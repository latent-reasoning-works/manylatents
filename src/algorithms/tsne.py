import numpy as np
from typing import Literal, Optional, Tuple

import torch
from lightning import LightningModule
from sklearn.manifold import TSNE
from torch import Tensor


class TSNEModule(LightningModule):
    def __init__(
        self,
        n_components: int = 2,
        datamodule=None,  # Reference to the datamodule if needed downstream
        network: Optional[torch.nn.Module] = None, # Unused for now; kept for interface consistency
        random_state: int = 42,
        perplexity: float = 30.0, 
        n_iter: int = 1000,
        learning_rate: int = 'auto',
        metric: str = "euclidean",
        fit_fraction: float = 1.0,  # Fraction of the first batch to use for fitting t-SNE
    ):
        super().__init__()
        self.n_components = n_components
        self.datamodule = datamodule
        self.random_state = random_state
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.fit_fraction = fit_fraction
        self._is_fitted = False
        self.model = TSNE(n_components=self.n_components, 
                          random_state=self.random_state, 
                          perplexity=self.perplexity,
                          n_iter=self.n_iter,
                          learning_rate=self.learning_rate,
                          metric=self.metric
                         )

    def fit_transform(self, x: Tensor) -> Tensor:
        """
        t-SNE does not support separate fit and transform, this function fit and transform on same fraction.
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
            embedding = self.model.fit_transform(x_np[:n_fit])
            self._is_fitted = True

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
        Applies t-SNE to the input batch and returns a dictionary containing:
          - a dummy loss (for compatibility with Lightning),
          - the t-SNE embedding,
          - the original labels.
        """
        x, y = batch
        embedding = self(x)
        return {"embedding": embedding}