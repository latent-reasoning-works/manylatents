import torch
from torch import Tensor
from typing import Optional, Union
import numpy as np
from .mds_algorithm import MultidimensionalScaling
from ..latent_module_base import LatentModule


class MDSModule(LatentModule):
    def __init__(
        self,
        ndim: int = 2,
        seed: Optional[int] = 42,
        how: str = "metric", #  choose from ['classic', 'metric', 'nonmetric']
        solver: str = "smacof", # choose from ["sgd", "smacof"]
        distance_metric: str = 'euclidean', # recommended values: 'euclidean' and 'cosine'
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(ndim, seed)
        self.fit_fraction = fit_fraction
        self.model = MultidimensionalScaling(ndim=ndim, 
                                            seed=seed,
                                            how=how,
                                            solver=solver,
                                            distance_metric=distance_metric,
                                            n_jobs=n_jobs,
                                            verbose=verbose)

    def fit(self, x: Tensor) -> None:
        """Fits MDS on all of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        emb = self.model.embed_MDS(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted MDS model. MDS can't be extend to new data, so we just return the embedding of the fitted data."""
        if not self._is_fitted:
            raise RuntimeError("MDS model is not fitted yet. Call `fit` first.")
        
        embedding = self.model.embedding
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)
    
    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()
        embedding = self.model.embed_MDS(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)