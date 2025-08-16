
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from ..latent_module_base import LatentModule


class PCAModule(LatentModule):
    def __init__(self, 
                 n_components: int = 2, 
                 random_state: int = 42, 
                 fit_fraction: float = 1.0,
                 **kwargs):
        super().__init__(n_components=n_components, 
                         init_seed=random_state,
                         **kwargs)
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=n_components,
                         random_state=random_state)
        self._is_fitted = False

    def fit(self, x: Tensor) -> None:
        """Fits PCA on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted PCA model."""
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)
    