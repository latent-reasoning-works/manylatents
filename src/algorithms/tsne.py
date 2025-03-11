
import torch
from torch import Tensor
from typing import Optional, Union
from openTSNE import TSNE

from .dimensionality_reduction import DimensionalityReductionModule


class TSNEModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: Optional[float] = 30.0,
        n_iter: Optional[int] = 1000, 
        learning_rate: Union[float, str] = 'auto',
        metric: str = "euclidean",
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(n_components, random_state, perplexity, n_iter, learning_rate, metric)
        self.fit_fraction = fit_fraction
        self.model = TSNE(n_components=self.n_components, 
                          random_state=self.random_state, 
                          perplexity=self.perplexity,
                          n_iter=self.n_iter,
                          learning_rate=self.learning_rate,
                          metric=self.metric
                         )

    def fit(self, x: Tensor) -> None:
        """Fits tsne on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """opentSNE support transform."""
        if not self._is_fitted:
            raise RuntimeError("tSNE model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)