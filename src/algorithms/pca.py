from typing import Literal, Optional, Tuple
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from lightning import LightningModule

class PCAModule(LightningModule):
    def __init__(
        self,
        n_components: int = 2,
        datamodule=None,  # Reference to the datamodule if needed downstream
        network: Optional[torch.nn.Module] = None,  # Unused for PCA; kept for interface consistency
        init_seed: int = 42,
        fit_fraction: float = 1.0,  # Fraction of the first batch to use for fitting PCA
    ):
        super().__init__()
        self.n_components = n_components
        self.datamodule = datamodule
        self.init_seed = init_seed
        self.fit_fraction = fit_fraction
        self.model = PCA(n_components=self.n_components)
        self._is_fitted = False

    def fit(self, x: Tensor) -> None:
        """
        Fits PCA on a subset of data.
        """
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """
        Transforms data using the fitted PCA model.
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Lightning. Wraps the transform method.
        """
        return self.transform(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning's training step. Fit PCA on the first batch if not already fitted.
        """
        x, y = batch
        if not self._is_fitted:
            self.fit(x)  # Fit on first batch (or fit outside trainer before)
        embedding = self(x)  # Calls forward
        self.log('train/embedding_mean', embedding.mean())
        return {"embedding": embedding}

    def validation_step(self, batch, batch_idx):
        """
        Validation step: only transform data, no fitting.
        """
        x, y = batch
        embedding = self(x)
        self.log('val/embedding_mean', embedding.mean())
        return {"embedding": embedding}

    def test_step(self, batch, batch_idx):
        """
        Test step: same as validation.
        """
        x, y = batch
        embedding = self(x)
        self.log('test/embedding_mean', embedding.mean())
        return {"embedding": embedding}
    