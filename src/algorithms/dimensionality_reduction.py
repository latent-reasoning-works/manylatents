from abc import ABC, abstractmethod

from torch import Tensor


class DimensionalityReductionModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42, fast_dev_run_dr: bool = False, n_samples_fast_dev: int = 100):
        """Base class for dimensionality reduction modules.
        
        Args:
            n_components: Number of components to reduce to
            init_seed: Random seed for reproducibility
            fast_dev_run_dr: If True, use only a small subset of data for fitting
            n_samples_fast_dev: Number of samples to use when fast_dev_run_dr is enabled
        """
        self.n_components = n_components
        self.init_seed = init_seed
        self.fast_dev_run_dr = fast_dev_run_dr
        self.n_samples_fast_dev = n_samples_fast_dev
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

    def _prepare_fit_data(self, x: Tensor) -> Tensor:
        """Prepare data for fitting, applying fast_dev_run_dr if enabled."""
        if self.fast_dev_run_dr:
            return x[:self.n_samples_fast_dev]
        return x
