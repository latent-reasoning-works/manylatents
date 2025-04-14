from abc import ABC, abstractmethod

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

    @property
    def affinity_matrix(self):
        """Fitted Affinity matrix. For topological metrics"""
        pass

    @property
    def kernel_matrix(self):
        """Fitted Kernel matrix. For topological metrics"""
        pass
