
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import resolve_backend, resolve_device, torchdr_knn_to_dense


class UMAPModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = 'euclidean',
        n_epochs: Optional[int] = 200,
        learning_rate: float = 1.0,
        fit_fraction: float = 1.0,
        backend: str | None = None,
        device: str | None = None,
        neighborhood_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device,
            neighborhood_size=neighborhood_size, **kwargs,
        )
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fit_fraction = fit_fraction
        self.random_state = random_state

        self._resolved_backend = resolve_backend(backend)
        self.model = self._create_model()

    def _create_model(self):
        if self._resolved_backend == "torchdr":
            from torchdr import UMAP

            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                device=resolve_device(self.device),
                random_state=self.random_state,
            )
        else:
            from umap import UMAP

            return UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
            )

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits UMAP on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            self.model.fit(x_torch)
        else:
            self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted UMAP model."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.transform(x_torch)
            return torch.tensor(embedding.detach().cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            embedding = self.model.transform(x_np)
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.fit_transform(x_torch)
            self._is_fitted = True
            return torch.tensor(embedding.detach().cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            embedding = self.model.fit_transform(x_np)
            self._is_fitted = True
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """Returns UMAP affinity matrix.

        UMAP's graph represents fuzzy membership strengths. This method can return
        either a row-stochastic (asymmetric) or symmetric diffusion operator version.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.
            use_symmetric: If True, return symmetric diffusion operator with guaranteed
                positive eigenvalues. If False, return row-stochastic matrix. Default False.

        Returns:
            N x N affinity matrix (row-normalized if use_symmetric=False, symmetric if True).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend == "torchdr":
                A = torchdr_knn_to_dense(self.model).cpu().numpy()
            else:
                A = np.asarray(self.model.graph_.todense())

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return A / row_sums

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Returns UMAP kernel matrix.

        For UMAP, the fuzzy simplicial set serves as both the kernel
        and affinity matrix. The graph already has zero diagonal.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.

        Returns:
            N x N kernel matrix (fuzzy simplicial set).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if self._resolved_backend == "torchdr":
            K = torchdr_knn_to_dense(self.model).cpu().numpy()
        else:
            K = np.asarray(self.model.graph_.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
