from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import resolve_backend, resolve_device, torchdr_knn_to_dense


class PHATEModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        knn: Optional[int] = 5,
        t: Union[int, str] = 15,  # Can be an integer or 'auto'
        decay: Optional[int] = 40,
        gamma: Optional[float] = 1,
        n_pca: Optional[int] = 100,
        n_landmark: Optional[int] = 2000,
        n_jobs: Optional[int] = -1,
        verbose=False,
        fit_fraction: float = 1.0,
        random_landmarking: bool = False,
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
        self.knn = neighborhood_size if neighborhood_size is not None else knn
        self.t = t
        self.decay = decay
        self.gamma = gamma
        self.n_pca = n_pca
        self.n_landmark = n_landmark
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fit_fraction = fit_fraction
        self.random_state = random_state
        self.random_landmarking = random_landmarking

        self._resolved_backend = resolve_backend(backend)
        self.model = self._create_model()

    def _create_model(self):
        if self._resolved_backend == "torchdr":
            from torchdr import PHATE

            # TorchDR PHATE param mapping: knn->k, decay->alpha
            # TorchDR PHATE does NOT support faiss/keops, force backend=None
            return PHATE(
                n_components=self.n_components,
                k=self.knn,
                t=self.t,
                alpha=self.decay,
                device=resolve_device(self.device),
                random_state=self.random_state,
                backend=None,  # TorchDR PHATE only supports None
            )
        else:
            from phate import PHATE

            # random_landmarking is supported in PHATE 2.0+
            phate_kwargs = {
                'n_components': self.n_components,
                'random_state': self.random_state,
                'knn': self.knn,
                't': self.t,
                'decay': self.decay,
                'gamma': self.gamma,
                'n_pca': self.n_pca,
                'n_landmark': self.n_landmark,
                'n_jobs': self.n_jobs,
                'verbose': self.verbose,
                'random_landmarking': self.random_landmarking,
            }
            return PHATE(**phate_kwargs)

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits PHATE on a subset of data."""
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
            # Store lightweight statistics and small sample for permutation detection in transform
            self._training_shape = x_np[:n_fit].shape
            self._training_mean = np.mean(x_np[:n_fit], axis=0)
            self._training_std = np.std(x_np[:n_fit], axis=0)
            # Store first 10 rows for identity checking (small memory footprint)
            self._training_sample = x_np[:min(10, n_fit)].copy()
            self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PHATE model."""
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.transform(x_torch)
            return torch.tensor(embedding.detach().cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            # Check for potential data permutation issues
            if (x_np.shape == self._training_shape and
                np.allclose(np.mean(x_np, axis=0), self._training_mean, rtol=1e-5) and
                np.allclose(np.std(x_np, axis=0), self._training_std, rtol=1e-5) and
                not np.array_equal(x_np[:len(self._training_sample)], self._training_sample)):

                import warnings
                warnings.warn(
                    "Transform data has identical shape and statistics to training data but are not identical. "
                    "This may indicate shuffled vs unshuffled versions of the same dataset. "
                    "Consider setting 'shuffle_traindata: false' in your data config to avoid PHATE warnings.",
                    UserWarning
                )
            embedding = self.model.transform(x_np)
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

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
            # Store lightweight statistics for permutation detection in transform
            self._training_shape = x_np[:n_fit].shape
            self._training_mean = np.mean(x_np[:n_fit], axis=0)
            self._training_std = np.std(x_np[:n_fit], axis=0)
            self._training_sample = x_np[:min(10, n_fit)].copy()
            embedding = self.model.fit_transform(x_np[:n_fit])
            self._is_fitted = True
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """
        Returns PHATE affinity matrix.

        PHATE's diffusion operator is row-stochastic (asymmetric). This method can return
        either the diffusion operator or a symmetric diffusion operator version.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
            use_symmetric: If True, return symmetric diffusion operator with guaranteed
                positive eigenvalues. If False, return row-stochastic diffusion operator. Default False.

        Returns:
            N x N affinity matrix (diffusion operator if use_symmetric=False, symmetric if True).
        """
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend == "torchdr":
                # TorchDR PHATE stores affinity_in_ as NxN dense in log-space
                A = torch.exp(self.model.affinity_in_.detach()).cpu().numpy()
            else:
                diff_op = self.model.diff_op
                A = np.asarray(diff_op)

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))
            return A

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns kernel matrix used to build diffusion operator.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N x N kernel matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")

        if self._resolved_backend == "torchdr":
            # TorchDR PHATE stores affinity_in_ as NxN dense in log-space
            K = torch.exp(self.model.affinity_in_.detach()).cpu().numpy()
        else:
            K = np.asarray(self.model.graph.K.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
