import logging
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
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
    
    original: NotRequired[np.ndarray]
    """Optional original data used for DR."""
    
    loss: NotRequired[Union[Tensor, float]]
    """Optional loss value from the training/validation step."""
    
    label: NotRequired[Tensor]
    """Optional labels for the embeddings, if available."""
    
class DimensionalityReductionCallback(BaseCallback, ABC):
    @abstractmethod
    def on_dr_end(self, dataset: Any, embeddings: np.ndarray) -> Any:
        """
        Called when the Dimensionality Reduction process is complete.
        
        Args:
            dataset (Any): A dataset object that exposes properties/methods 
                           like `full_data` and `get_labels()`.
            embeddings (np.ndarray): The computed embeddings.
        
        Returns:
            Any: A result, for example a file path or a dictionary of metrics.
        """
        pass