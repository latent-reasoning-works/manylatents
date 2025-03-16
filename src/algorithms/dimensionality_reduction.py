from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch import Tensor


class DimensionalityReductionModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42):
        """Base class for dimensionality reduction modules."""
        self.n_components = n_components
        self.init_seed = init_seed
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor) -> None:
        """Fit the dimensionality reduction model to the data."""
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """Transform data using the fitted model."""
        pass

    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit and then transform data."""
        self.fit(x)
        return self.transform(x)
    
    def _compute_correlation(self, original_x: torch.Tensor, embeddings: torch.Tensor) -> float:
        """
        Compute the Pearson correlation between the pairwise distances of the original data
        and the embeddings.
        """
        # Convert tensors to numpy arrays
        orig_np = original_x.detach().cpu().numpy()
        emb_np = embeddings.detach().cpu().numpy()
        # Compute pairwise distances
        orig_dists = pdist(orig_np)
        emb_dists = pdist(emb_np)
        # Compute and return the Pearson correlation coefficient
        corr = np.corrcoef(orig_dists, emb_dists)[0, 1]
        return corr
    
    
    def evaluate(self, original_x: torch.Tensor, embeddings: np.array) -> dict:
        """
        Default evaluation that returns general DR metrics.
        Child classes can override this to compute module-specific metrics.
        
        Args:
            original_x: The original high-dimensional data tensor.
            embeddings: The low-dimensional embeddings (as a NumPy array or Tensor).
            
        Returns:
            A dictionary mapping metric names to their computed values.
            For example: {"correlation": 0.92}
        """
        correlation = self._compute_correlation(original_x, embeddings)
        # You can extend this dictionary with additional metrics if needed.
        return {"correlation": correlation}