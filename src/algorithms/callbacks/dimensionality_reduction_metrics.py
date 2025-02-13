import torch
from torch import Tensor


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

class DimensionalityReductionMetricsCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
    pass