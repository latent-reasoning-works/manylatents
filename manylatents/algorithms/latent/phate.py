import warnings
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import check_torchdr_available, resolve_device


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
        self._resolved_backend = self._resolve_phate_backend(backend)
        self.model = self._create_model()

    @staticmethod
    def _resolve_phate_backend(backend: str | None) -> str | None:
        if backend is None or backend == "sklearn":
            return None
        if backend == "torchdr":
            warnings.warn(
                "backend='torchdr' is kept for compatibility. "
                "Prefer backend='gpu_phate' for GPU PHATE.",
                UserWarning,
            )
            return "torchdr"
        if backend == "gpu_phate":
            return "gpu_phate"
        if backend == "auto":
            # Auto policy: prefer local GPU PHATE over TorchDR PHATE.
            return "gpu_phate" if torch.cuda.is_available() else None
        raise ValueError(
            f"Unknown backend: {backend!r}. Use None, 'sklearn', 'torchdr', 'gpu_phate', or 'auto'."
        )

    def _create_model(self):
        if self._resolved_backend == "torchdr":
            if not check_torchdr_available():
                raise ImportError(
                    "PHATE backend='torchdr' requested but torchdr is not installed. "
                    "Install with: pip install manylatents[torchdr] "
                    "or use backend='gpu_phate'."
                )
            if self.random_landmarking:
                warnings.warn(
                    "backend='torchdr' does not support random landmarking; "
                    "random_landmarking is ignored.",
                    UserWarning,
                )
            from torchdr import PHATE

            return PHATE(
                n_components=self.n_components,
                k=self.knn,
                t=self.t,
                alpha=self.decay,
                device=resolve_device(self.device),
                random_state=self.random_state,
                backend=None,
            )
        if self._resolved_backend == "gpu_phate":
            from .gpu_phate_local import PHATE

            return PHATE(
                n_components=self.n_components,
                k=self.knn,
                t=self.t,
                alpha=self.decay,
                n_landmarks=self.n_landmark,
                random_landmarking=self.random_landmarking,
                device=resolve_device(self.device),
                random_state=self.random_state,
                backend=None,
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

    def fit(self, x, y=None) -> None:
        """Fits PHATE on a subset of data."""
        x_np = _to_numpy(x)
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))

        if self._resolved_backend in {"torchdr", "gpu_phate"}:
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

    def transform(self, x):
        """Transforms data using the fitted PHATE model."""
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")

        x_np = _to_numpy(x)

        if self._resolved_backend in {"torchdr", "gpu_phate"}:
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding_np = self.model.transform(x_torch).detach().cpu().numpy()
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
            embedding_np = self.model.transform(x_np)
        return _to_output(embedding_np, x)

    def fit_transform(self, x, y=None):
        """Fit and then transform on same data."""
        x_np = _to_numpy(x)
        n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))

        if self._resolved_backend in {"torchdr", "gpu_phate"}:
            import torch as th
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding_np = self.model.fit_transform(x_torch).detach().cpu().numpy()
            self._is_fitted = True
        else:
            # Store lightweight statistics for permutation detection in transform
            self._training_shape = x_np[:n_fit].shape
            self._training_mean = np.mean(x_np[:n_fit], axis=0)
            self._training_std = np.std(x_np[:n_fit], axis=0)
            self._training_sample = x_np[:min(10, n_fit)].copy()
            embedding_np = self.model.fit_transform(x_np[:n_fit])
            self._is_fitted = True
        return _to_output(embedding_np, x)

    def affinity(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
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
            K = self.kernel(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend in {"torchdr", "gpu_phate"}:
                if hasattr(self.model, "diff_op_"):
                    A = self.model.diff_op_.detach().cpu().numpy()
                else:
                    raise NotImplementedError(
                        "affinity(use_symmetric=False) is not available for backend='torchdr' "
                        "when diff_op_ is not exposed."
                    )
            else:
                diff_op = self.model.diff_op
                A = np.asarray(diff_op)

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))
            return A

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns kernel matrix used to build diffusion operator.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N x N kernel matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")

        if self._resolved_backend in {"torchdr", "gpu_phate"}:
            if hasattr(self.model, "kernel_"):
                K = self.model.kernel_.detach().cpu().numpy()
            else:
                raise NotImplementedError(
                    "kernel() is not available for backend='torchdr' when kernel_ is not exposed."
                )
        else:
            K = np.asarray(self.model.graph.K.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
