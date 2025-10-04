
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from ..latent_module_base import LatentModule


class PCAModule(LatentModule):
    def __init__(self,
                 n_components: int = 2,
                 random_state: int = 42,
                 fit_fraction: float = 1.0,
                 **kwargs):
        super().__init__(n_components=n_components,
                         init_seed=random_state,
                         **kwargs)
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=n_components,
                         random_state=random_state)
        self._is_fitted = False
        self._fit_data = None  # Store fitted data for covariance computation

    def fit(self, x: Tensor) -> None:
        """Fits PCA on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])

        # Store fitted data for covariance computation
        self._fit_data = x_np[:n_fit]
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PCA model."""
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns sample covariance matrix (Gram matrix).

        PCA works with the covariance structure. The kernel matrix is the
        Gram matrix K = X_centered @ X_centered.T, which represents the
        covariance between samples.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N Gram matrix (sample covariance).
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")

        # Center the data
        X_centered = self._fit_data - self.model.mean_

        # Compute Gram matrix: K = X @ X.T
        K = X_centered @ X_centered.T

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))

        return K

    def affinity_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns normalized covariance matrix (Gram matrix / (n-1)).

        For PCA, the affinity matrix is the sample covariance matrix, normalized
        so that eigenvalues match the variance explained by principal components.
        This is K / (n-1) where K is the Gram matrix.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N normalized covariance matrix (eigenvalues = variance explained).
        """
        K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
        n = self._fit_data.shape[0]
        return K / (n - 1)
    