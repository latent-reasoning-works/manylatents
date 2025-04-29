import torch
from torch import Tensor
from typing import Optional, Union

from src.utils.diffusion_map import DiffusionMap
from .dimensionality_reduction import DimensionalityReductionModule

class DiffusionMapModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        knn: Optional[int] = 5,
        t: Union[int, str] = 15, # Can be an integer or 'auto'
        decay: Optional[int] = 40,
        n_pca: Optional[int] = None,
        n_landmark: Optional[int] = 2000,
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
    ):
        super().__init__(n_components, random_state)
        self.fit_fraction = fit_fraction
        self.model = DiffusionMap(n_components=n_components, 
                                  random_state=random_state,
                                  knn=knn,
                                  t=t,
                                  decay=decay,
                                  n_pca=n_pca,
                                  n_landmark=n_landmark,
                                  n_jobs=n_jobs,
                                  verbose=verbose)

    def fit(self, x: Tensor) -> None:
        """Fits DiffusionMap on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted DiffusionMap model."""
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    @property
    def affinity_matrix(self):
        """Returns diffusion operator, without diagonal"""
        diff_op = self.model.diff_op 
        A = diff_op - np.diag(diff_op)*np.eye(len(diff_op))
        return A

    @property
    def kernel_matrix(self):
        """Returns kernel matrix used to build diffusion operator"""
        K =  np.asarray(self.model.graph.K.todense())
        K = K - np.diag(K)*np.eye(len(K))
        return K