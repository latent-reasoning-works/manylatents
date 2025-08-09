from torch import Tensor

from src.algorithms.latent_module_base import LatentModule


class NoOpModule(LatentModule):
    def __init__(self):
        super().__init__(n_components=0, init_seed=42)
        self._is_fitted = True 

    def fit(self, x: Tensor) -> None:
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("NoOp module is not fitted. Call `fit` first.")
        return x
