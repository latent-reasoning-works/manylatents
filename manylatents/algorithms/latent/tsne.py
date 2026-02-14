from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import resolve_backend, resolve_device, torchdr_knn_to_dense


def build_dense_distance_matrix(distances, neighbors) -> np.ndarray:
    """
    Construct a full NxN matrix from distances and neighbors.

    Args:
        distances: NxK array of distances to neighbors
        neighbors:  NxK indices of neighbors

    Returns:
        NxN NumPy array with distances filled in, zeros elsewhere.
    """

    N = neighbors.shape[0]
    matrix = np.zeros((N, N), dtype=distances.dtype)

    for i in range(N):
        matrix[i, neighbors[i]] = distances[i]

    return matrix


class TSNEModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: Optional[float] = 30.0,
        n_iter_early: Optional[int] = 250,
        n_iter_late: Optional[int] = 750,
        learning_rate: Union[float, str] = 'auto',
        metric: str = "euclidean",
        initialization: str = "random",
        fit_fraction: float = 1.0,
        backend: str | None = None,
        device: str | None = None,
        **kwargs
    ):
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device, **kwargs,
        )
        self.perplexity = perplexity
        self.n_iter_early = n_iter_early
        self.n_iter_late = n_iter_late
        self.learning_rate = learning_rate
        self.metric = metric
        self.initialization = initialization
        self.fit_fraction = fit_fraction
        self.random_state = random_state

        self._resolved_backend = resolve_backend(backend)
        if self._resolved_backend == "torchdr":
            self.model = self._create_torchdr_model()

    def _create_torchdr_model(self):
        from torchdr import TSNE

        return TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            device=resolve_device(self.device),
            random_state=self.random_state,
        )

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        x_fit = x_np[:n_fit]

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_fit).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            self.model.fit(x_torch)
        else:
            from openTSNE.affinity import PerplexityBasedNN
            from openTSNE.tsne import TSNEEmbedding

            # monkey patch to allow for large perplexity
            # Overwriting:
            # https://github.com/pavlin-policar/openTSNE/blob/52ae1d67cbe2b99995e6c8dc0fcc3992344998bc/openTSNE/affinity.py#L340
            def do_nothing_check_perplexity(perplexity, k_neighbors):
                # Always just return the perplexity passed in, no checks or clamping
                return perplexity
            PerplexityBasedNN.check_perplexity = staticmethod(do_nothing_check_perplexity)

            self.affinities = PerplexityBasedNN(
                x_fit,
                perplexity=self.perplexity,
                metric=self.metric,
                n_jobs=-1,
                method="approx",
                random_state=self.random_state
            )

            init = _get_tsne_initialization(
                init_arg=self.initialization,
                x_fit=x_fit,
                n_fit=n_fit,
                n_components=self.n_components,
                random_state=self.random_state,
                affinities=self.affinities,
            )

            self.embedding_train = TSNEEmbedding(
                init, self.affinities, random_state=self.random_state
            )

            self.embedding_train.optimize(
                n_iter=self.n_iter_early,
                learning_rate=self.learning_rate,
                exaggeration=12,
                momentum=0.5,
                inplace=True
            )

            self.embedding_train.optimize(
                n_iter=self.n_iter_late,
                learning_rate=self.learning_rate,
                momentum=0.8,
                inplace=True
            )

        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("tSNE model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.transform(x_torch)
            return torch.tensor(embedding.detach().cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            embedding_out = self.embedding_train.transform(x_np)
            return torch.tensor(embedding_out, device=x.device, dtype=x.dtype)

    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()
        n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.fit_transform(x_torch)
            self._is_fitted = True
            return torch.tensor(embedding.detach().cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            self.fit(x, y)
            return torch.tensor(np.array(self.embedding_train), device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """
        Returns t-SNE affinity matrix.

        openTSNE's P matrix is symmetrized and scaled by 1/(2N). This method can return
        either a row-stochastic (asymmetric) or symmetric diffusion operator version.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: P matrix typically has very small diagonal values already.
            use_symmetric: If True, return symmetric diffusion operator with guaranteed
                positive eigenvalues. If False, return row-stochastic matrix. Default False.

        Returns:
            N x N affinity matrix (row-normalized if use_symmetric=False, symmetric if True).
        """
        if not self._is_fitted:
            raise RuntimeError("t-SNE model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend == "torchdr":
                A = torchdr_knn_to_dense(self.model).cpu().numpy()
            else:
                A = np.asarray(self.affinities.P.todense())

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))

            # Row-normalize to make it a proper transition matrix
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return A / row_sums

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns Gaussian kernel matrix built from raw knn distances.

        This constructs a dense kernel from the perplexity-calibrated Gaussian
        similarities computed during t-SNE's affinity computation.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N x N kernel matrix based on Gaussian similarities.
        """
        if not self._is_fitted:
            raise RuntimeError("t-SNE model is not fitted yet. Call `fit` first.")

        if self._resolved_backend == "torchdr":
            K = torchdr_knn_to_dense(self.model).cpu().numpy()
        else:
            K = np.asarray(self.affinities.P.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))

        return K


def _get_tsne_initialization(
    init_arg: Union[str, np.ndarray],
    x_fit: np.ndarray,
    n_fit: int,
    n_components: int,
    random_state: Optional[int],
    affinities=None,
) -> np.ndarray:
    """
    Return a (n_fit x n_components) array for openTSNE initialization.
    init_arg can be:
      - "pca", "random", "spectral", "rescale", "jitter"
      - any custom ndarray of shape (n_fit, n_components)
    """
    from openTSNE import initialization

    # custom array takes absolute precedence
    if isinstance(init_arg, np.ndarray):
        return init_arg

    method = init_arg.lower()
    if method == "pca":
        return initialization.pca(
            X=x_fit,
            n_components=n_components,
            random_state=random_state,
            add_jitter=True,
        )

    if method == "random":
        return initialization.random(
            n_samples=n_fit,
            n_components=n_components,
            random_state=random_state,
        )

    if method == "spectral":
        # affinities.P is the sparse adjacency / transition matrix
        return initialization.spectral(
            A=affinities.P,
            n_components=n_components,
            random_state=random_state,
            add_jitter=True,
        )

    if method == "rescale":
        base = initialization.pca(
            X=x_fit,
            n_components=n_components,
            random_state=random_state,
            add_jitter=False,
        )
        return initialization.rescale(x=base, target_std=0.0001)

    if method == "jitter":
        base = initialization.pca(
            X=x_fit,
            n_components=n_components,
            random_state=random_state,
            add_jitter=False,
        )
        return initialization.jitter(
            x=base,
            scale=0.01,
            random_state=random_state,
        )

    raise ValueError(
        f"Unsupported TSNE initialization `{init_arg}`; "
        "choose one of "
        "['pca','random','spectral','rescale','jitter'] or pass an ndarray"
    )
