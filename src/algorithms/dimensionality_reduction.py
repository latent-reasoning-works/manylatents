from abc import ABC, abstractmethod
from typing import Optional, Tuple, Literal
import torch
from torch import Tensor
from lightning import LightningModule

class DimensionalityReductionModule(LightningModule, ABC):
    def __init__(self, n_components: int = 2, init_seed: int = 42):
        """Base class for dimensionality reduction modules."""
        super().__init__()
        self.n_components = n_components
        self.init_seed = init_seed
        self._is_fitted = False

    @abstractmethod
    def fit(self, x: Tensor) -> None:
        """Fit the dimensionality reduction model to the data."""
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """Transform data using the fitted model."""
        pass

    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit and then transform data. Avoid using in production pipelines."""
        self.fit(x)
        return self.transform(x)

    def forward(self, x: Tensor) -> Tensor:
        """Use transform for inference."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling forward.")
        return self.transform(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def shared_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, phase: Literal["train", "val", "test"]):
        x, y = batch
        embedding = self(x)
        self.log(f"{phase}/embedding_mean", embedding.mean())
        return {"embedding": embedding, "labels": y}
