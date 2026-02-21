import torch
from torch import Tensor

from .latent_module_base import LatentModule


class ArchetypalAnalysisModule(LatentModule):
    def __init__(
        self,
        n_components: int = 3,
        method: str = "pgd",
        max_iter: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(n_components=n_components, 
                         init_seed=random_state,
                         **kwargs)
        self.method = method
        self.max_iter = max_iter
        self.model = None
        self._is_fitted = False

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits the archetypal analysis model to the input data."""
        from archetypes import AA

        x_np = x.detach().cpu().numpy()
        method_kwargs = {"max_iter_optimizer": self.max_iter}

        self.model = AA(
            n_archetypes=self.n_components,
            method=self.method,
            method_kwargs=method_kwargs,
            random_state=self.init_seed,
        )
        self.model.fit(x_np)
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms the input into similarity degrees to each archetype."""
        if not self._is_fitted:
            raise RuntimeError("Archetypal Analysis model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        similarity = self.model.transform(x_np)
        return torch.tensor(similarity, device=x.device, dtype=x.dtype)
