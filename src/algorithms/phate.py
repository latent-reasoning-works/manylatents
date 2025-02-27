
import torch
from torch import Tensor
from typing import Optional, Union
from phate import PHATE

from .dimensionality_reduction import DimensionalityReductionModule


class PHATEModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        knn: Optional[int] = 5,
        t: Union[int, str] = 15, # Can be an integer or 'auto'
        decay: Optional[int] = 40,
        gamma: Optional[float] = 1, 
        n_pca: Optional[int] = None,
        n_landmark: Optional[int] = 2000,
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(n_components, random_state, knn, t, decay, gamma, n_pca, n_landmark, n_jobs, verbose)
        self.fit_fraction = fit_fraction
        self.model = PHATE(n_components=self.n_components, 
                           random_state=self.random_state,
                           knn=self.knn,
                           t=self.t,
                           decay=self.decay,
                           gamma=self.gamma,
                           n_pca=self.n_pca,
                           n_landmark=self.n_landmark,
                           n_jobs=self.n_jobs,
                           verbose=self.verbose)

    def fit(self, x: Tensor) -> None:
        """Fits PHATE on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PHATE model."""
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)