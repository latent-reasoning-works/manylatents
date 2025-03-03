from abc import ABC, abstractmethod

from torch import Tensor
import numpy as np

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
    
    def evaluate(self, embeddings:np.array) -> float:
        """
        Compute a metric for this DR module.
        Child classes can override this to compute, e.g.,
        - trustworthiness
        - etc.
        If not overriden, returns a dictionary with an error key.
        """
        return {"error": 0.0}
