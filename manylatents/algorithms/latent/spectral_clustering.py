import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import SpectralClustering
from typing import Optional, Union
from .latent_module_base import LatentModule

class SpectralClusteringModule(LatentModule):
    def __init__(
        self,
        n_clusters: int = 5,
        eigen_solver: Optional[str] = None,
        n_components: Optional[int] = None,
        random_state: Optional[int] = None,
        n_init: int = 10,
        gamma: float = 1.0,
        affinity: str = 'precomputed',
        n_neighbors: int = 10,
        eigen_tol: Union[float, str] = 'auto',
        assign_labels: str = 'kmeans',
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(init_seed=random_state)
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            n_components=n_components,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self._is_fitted = False

    def fit(self, x: Union[Tensor, np.ndarray]) -> None:
        x_np = x.detach().cpu().numpy() if isinstance(x, Tensor) else x
        self.model.fit(x_np)
        self._is_fitted = True

    def transform(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("SpectralClustering model is not fitted yet. Call `fit` first.")
        # SpectralClustering does not support out-of-sample prediction, so only works for fitted data
        return torch.tensor(self.model.labels_, dtype=torch.long)

    def fit_transform(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        x_np = x.detach().cpu().numpy() if isinstance(x, Tensor) else x
        labels = self.model.fit_predict(x_np)
        self._is_fitted = True
        return torch.tensor(labels, dtype=torch.long)
