from abc import ABC, abstractmethod

from torch import Tensor


class LatentModule(ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42, **kwargs):
        """Base class for latent modules (DR, clustering, etc.)."""
        self.n_components = n_components
        self.init_seed = init_seed
        # Flexible handling: if datamodule is passed, store it as a weak port
        self.datamodule = kwargs.pop('datamodule', None)
        # Ignore any other unexpected kwargs to maintain compatibility
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor) -> None:
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        pass

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)
