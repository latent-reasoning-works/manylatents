import torch
from torch import Tensor
from typing import Optional, Union
import numpy as np

from ..diffusionmap_algorithm import DiffusionMap
from ..latent_module_base import LatentModule

class DiffusionMapModule(LatentModule):
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
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
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

    def affinity_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns diffusion operator.

        The diffusion operator is computed during fit and represents the
        transition matrix of the diffusion process.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N diffusion operator matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")

        # Recompute diffusion operator from kernel
        K = np.asarray(self.model.G.kernel.todense())
        from ..diffusionmap_algorithm import compute_dm
        _, _, diff_op, _ = compute_dm(K, alpha=1.0)

        if ignore_diagonal:
            diff_op = diff_op - np.diag(np.diag(diff_op))
        return diff_op

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns kernel matrix used to build diffusion operator.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N kernel matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")

        K = np.asarray(self.model.G.kernel.todense())
        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K