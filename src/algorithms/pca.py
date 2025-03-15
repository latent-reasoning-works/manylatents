from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from torch import Tensor

from .dimensionality_reduction import DimensionalityReductionModule


class PCAModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        init_seed: int = 42,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(n_components, init_seed)
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=self.n_components)

    def fit(self, x: Tensor) -> None:
        """Fits PCA on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PCA model."""
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)
    
    def _compute_correlation(self, x: Tensor, embeddings: Tensor) -> float:
        """
        Compute the Pearson correlation between the pairwise distances in the 
        original data and the PCA-transformed embedding.
        """
        x_np = x.detach().cpu().numpy()
        embeddings_np = embeddings.detach().cpu().numpy()
        orig_dists = pdist(x_np)
        pca_dists = pdist(embeddings_np)
        corr = np.corrcoef(orig_dists, pca_dists)[0, 1]
        return corr
    
    def evaluate(self, x: Tensor, embeddings: Optional[Tensor] = None) -> Dict[str, float]:
        """
        Compute PCA-specific metrics.
        
        Args:
            x: Original high-dimensional data tensor.
            embeddings: (Optional) Precomputed PCA embeddings. 
                        If not provided, it will be computed from x.
        
        Returns:
            A dictionary with metric names as keys and computed values as floats.
            For example: {'pca_correlation': 0.95, 'variance_explained': 0.85}
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Cannot evaluate.")
        
        # Use precomputed embeddings if available, otherwise compute them.
        if embeddings is None:
            embeddings = self.transform(x)
        
        # Compute PCA-specific metrics
        pca_corr = self._compute_correlation(x, embeddings)
        
        # If you add more PCA metrics, compute them here.
        # For example:
        # variance_explained = self._compute_variance_explained(x, embeddings)
        
        # Aggregate metrics into a dictionary.
        metrics = {
            "pca_correlation": pca_corr,
            # "variance_explained": variance_explained,
        }
        return metrics
