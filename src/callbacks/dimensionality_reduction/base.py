import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import NotRequired, Required, TypedDict

from src.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

class DimensionalityReductionOutputs(TypedDict, total=False):
    """
    Defines the minimal output contract for DR models.
    All DR LightningModules should return at least an 'embeddings' tensor,
    along with optional 'loss' and 'label' entries.
    """
    embeddings: Required[Tensor]
    """Reduced embeddings from the model, e.g. PHATE, UMAP, PCA."""
    
    loss: NotRequired[Union[torch.Tensor, float]]
    """Optional loss value from the training/validation step."""
    
    label: NotRequired[Tensor]
    """Optional labels for the embeddings, if available."""
    
class DimensionalityReductionCallback(BaseCallback, ABC):
    @abstractmethod
    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Called when the Dimensionality Reduction process is complete.
        
        Args:
            embeddings (np.ndarray): The computed embeddings.
            labels (np.ndarray, optional): Optional labels associated with the embeddings.
        """
        pass
