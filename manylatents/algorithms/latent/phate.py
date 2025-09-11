
from typing import Optional, Union

import numpy as np
import torch
from phate import PHATE
from torch import Tensor

from ..latent_module_base import LatentModule


class PHATEModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        knn: Optional[int] = 5,
        t: Union[int, str] = 15, # Can be an integer or 'auto'
        decay: Optional[int] = 40,
        gamma: Optional[float] = 1, 
        n_pca: Optional[int] = 100,
        n_landmark: Optional[int] = 2000,
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.model = PHATE(n_components=n_components, 
                           random_state=random_state,
                           knn=knn,
                           t=t,
                           decay=decay,
                           gamma=gamma,
                           n_pca=n_pca,
                           n_landmark=n_landmark,
                           n_jobs=n_jobs,
                           verbose=verbose)

    def fit(self, x: Tensor) -> None:
        """Fits PHATE on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        
        # Store lightweight statistics and small sample for permutation detection in transform
        self._training_shape = x_np[:n_fit].shape
        self._training_mean = np.mean(x_np[:n_fit], axis=0)
        self._training_std = np.std(x_np[:n_fit], axis=0)
        # Store first 10 rows for identity checking (small memory footprint)
        self._training_sample = x_np[:min(10, n_fit)].copy()
        
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PHATE model."""
        if not self._is_fitted:
            raise RuntimeError("PHATE model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        
        # Check for potential data permutation issues
        if (x_np.shape == self._training_shape and 
            np.allclose(np.mean(x_np, axis=0), self._training_mean, rtol=1e-5) and
            np.allclose(np.std(x_np, axis=0), self._training_std, rtol=1e-5) and
            not np.array_equal(x_np[:len(self._training_sample)], self._training_sample)):
            
            import warnings
            warnings.warn(
                "Transform data has identical shape and statistics to training data but are not identical. "
                "This may indicate shuffled vs unshuffled versions of the same dataset. "
                "Consider setting 'shuffle_traindata: false' in your data config to avoid PHATE warnings.",
                UserWarning
            )
        
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
