import warnings

import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torchmetrics import RandScore
from typing_extensions import NotRequired, Required, TypedDict, override


class DimensionalityReductionOutputs(TypedDict, total=False):
    """
    Minimally returned output from the training/val Dimensionality Reduction
    LightningModules so that metrics can be automatically casted into the 
    DimensionalityReductionMetricsCallback.
    """
    embeddings: Required[Tensor]
    """Reduced embeddings from the model, e.g. PHATE, UMAP, PCA."""
    
    loss: NotRequired[torch.Tensor | float]
    
    """Loss value from the training/val step."""
    
    label: NotRequired[Tensor]
    """Labels for the embeddings, if available."""

class DimensionalityReductionMetricsCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

        self.embeddings_buffer = []
        self.labels_buffer = []
        
    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: DimensionalityReductionOutputs,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "embeddings" in outputs:
            self.embeddings_buffer.append(outputs["embeddings"].detach().cpu())
        if "label" in outputs:
            self.labels_buffer.append(outputs["label"].detach().cpu())

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.embeddings_buffer:
            return
        all_embeddings = torch.cat(self.embeddings_buffer, dim=0)
        try:
            emb_np = all_embeddings.numpy()
            rand_score = RandScore(emb_np)
            pl_module.log("val/RandScore", rand_score, prog_bar=True)
        except Exception as e:
            warnings.warn(f"Could not compute RandScore score: {e}")
        # Clear buffers for the next epoch
        self.embeddings_buffer.clear()
        self.labels_buffer.clear()
