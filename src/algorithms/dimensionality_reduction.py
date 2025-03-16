from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from src.metrics.correlation import PearsonCorrelation
from src.metrics.trustworthiness import Trustworthiness


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
    
    def evaluate(self, original_x: torch.Tensor, embeddings: np.array) -> dict:
        """
        Default evaluation that returns general DR metrics.
        Child classes can override this to compute module-specific metrics.
        
        Args:
            original_x: The original high-dimensional data tensor.
            embeddings: The low-dimensional embeddings (as a NumPy array or Tensor).
            
        Returns:
            A dictionary mapping metric names to their computed values.
            For example: {"pearson_correlation": 0.92}
        """
        pearson_correlation = PearsonCorrelation(original_x, embeddings)
        trustworthiness = Trustworthiness(original_x, embeddings)
        # You can extend this dictionary with additional metrics if needed.
        return {"correlation": pearson_correlation,
                "trustworthiness": trustworthiness}