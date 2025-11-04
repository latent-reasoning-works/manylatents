from torch import Tensor

from manylatents.algorithms.latent.latent_module_base import LatentModule


class NoOpModule(LatentModule):
    def __init__(self, **kwargs):
        super().__init__(n_components=0, init_seed=42, **kwargs)
        self._is_fitted = True 

    def fit(self, x: Tensor) -> None:
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("NoOp module is not fitted. Call `fit` first.")
        return x
