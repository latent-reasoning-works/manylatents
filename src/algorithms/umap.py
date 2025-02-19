from typing import Literal, Optional, Tuple
import torch
from lightning import LightningModule
from umap import UMAP
from torch import Tensor


class UMAPModule(LightningModule):
    def __init__(
        self,
        n_components: int = 2,
        datamodule=None,  # Reference to the datamodule if needed downstream
        network: Optional[torch.nn.Module] = None,  # Unused for PCA; kept for interface consistency
        random_state: int = 42,
        n_neighbors: int = 15,
        n_epochs: int = 500,
        fit_fraction: float = 1.0,  # Fraction of the first batch to use for fitting PCA
    ):
        super().__init__()
        self.n_components = n_components
        self.datamodule = datamodule
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.fit_fraction = fit_fraction
        self._is_fitted = False
        self.model = UMAP(
                            n_components=self.n_components, 
                            random_state=self.random_state, 
                            n_neighbors=self.n_neighbors, 
                            n_epochs=self.n_epochs,
                          )

    def fit_transform(self, x: Tensor) -> Tensor:
        """
        Applies UMAP transformation to the input tensor x.
        On the first call, fits UMAP using only a fraction of the batch (if specified).
        """
        if isinstance(x, torch.Tensor):
            device = x.device
            dtype = x.dtype
            x_np = x.detach().cpu().numpy()
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            x_np = x

        if not self._is_fitted:
            n_samples = x_np.shape[0]
            # Use only a fraction of the first batch for fitting.
            n_fit = max(1, int(self.fit_fraction * n_samples))
            self.model.fit(x_np[:n_fit])
            self._is_fitted = True
        # Transform the full batch using the fitted UMAP model.
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=device, dtype=dtype)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")
    
    def shared_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        phase: Literal["train", "val", "test"],
    ):
        """
        Applies UMAP to the input batch and returns a dictionary containing:
          - a dummy loss (for compatibility with Lightning),
          - the UMAP embedding,
          - the original labels.
        """
        x, y = batch
        embedding = self(x)
        return {"embedding": embedding}