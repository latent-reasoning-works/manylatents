
from typing import Optional

import torch
from torch import Tensor
from umap import UMAP

from .dimensionality_reduction import DimensionalityReductionModule


class UMAPModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        n_epochs: Optional[int] = 200,
        learning_rate: float = 1.0,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(n_components, random_state)
        self.fit_fraction = fit_fraction
        self.model = UMAP(n_components=n_components, 
                           random_state=random_state,
                           n_neighbors=n_neighbors,
                           metric=metric,
                           n_epochs=n_epochs,
                           learning_rate=learning_rate)

    def fit(self, x: Tensor) -> None:
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
    
    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()
        embedding = self.model.fit_transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)