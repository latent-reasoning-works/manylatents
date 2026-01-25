
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from umap import UMAP

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator


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
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.model = UMAP(n_components=n_components, 
                           random_state=random_state,
                           n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           metric=metric,
                           n_epochs=n_epochs,
                           learning_rate=learning_rate)

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits UMAP on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted UMAP model. Transform is only used when x is new data (not fitted data)."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)
    
    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()
        embedding = self.model.fit_transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """
        Returns UMAP affinity matrix.

        UMAP's graph represents fuzzy membership strengths. This method can return
        either a row-stochastic (asymmetric) or symmetric diffusion operator version.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.
            use_symmetric: If True, return symmetric diffusion operator with guaranteed
                positive eigenvalues. If False, return row-stochastic matrix. Default False.

        Returns:
            N×N affinity matrix (row-normalized if use_symmetric=False, symmetric if True).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            # Return symmetric diffusion operator for positive eigenvalue guarantee
            K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            # Return row-stochastic matrix (original behavior)
            A = np.asarray(self.model.graph_.todense())
            if ignore_diagonal:
                A = A - np.diag(np.diag(A))

            # Row-normalize to make it a proper transition matrix
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            A_normalized = A / row_sums

            return A_normalized

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns UMAP kernel matrix (same as graph_ for UMAP).

        For UMAP, the fuzzy simplicial set serves as both the kernel
        and affinity matrix. The graph already has zero diagonal.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.

        Returns:
            N×N kernel matrix (fuzzy simplicial set).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        K = np.asarray(self.model.graph_.todense())
        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
