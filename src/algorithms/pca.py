from typing import Optional

import torch
from sklearn.decomposition import PCA
from torch import Tensor

from .dimensionality_reduction import DimensionalityReductionModule


class PCAModule(DimensionalityReductionModule):
    def __init__(self, n_components: int = 2, 
                 init_seed: int = 42, 
                 fit_fraction: float = 1.0):
        super().__init__(n_components, init_seed)
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=self.n_components)
        self._is_fitted = False

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
    
    def evaluate(self, original_x: torch.Tensor, embeddings: Optional[torch.Tensor] = None) -> dict:
        """
        Compute PCA-specific metrics by extending the general DR metrics.
        
        Args:
            original_x: The original high-dimensional data tensor.
            embeddings: Optional precomputed PCA embeddings. If not provided,
                        they are computed via self.transform(original_x).
                        
        Returns:
            A tuple of (error_metric, metrics_dict), where error_metric might be 
            the pca_correlation and metrics_dict contains additional metrics.
        """
        if embeddings is None:
            embeddings = self.transform(original_x)

        # Call the parent's evaluate method to compute general DR metrics,
        # e.g., the default correlation metric.
        metrics = super().evaluate(original_x, embeddings)

        # Extend the metrics with PCA-specific evaluations.
        # For example, compute the total variance explained by the selected components.
        variance_explained = self.model.explained_variance_ratio_.sum() if self._is_fitted else None
        
        # Add additional PCA metrics as needed.
        metrics.update({"pca_variance_explained": variance_explained})
        
        return metrics
