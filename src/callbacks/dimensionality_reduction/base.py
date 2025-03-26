import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from torch import Tensor
from typing_extensions import NotRequired, TypedDict

from src.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

class DimensionalityReductionOutputs(TypedDict, total=False):
    """
    Defines the minimal output contract for DR models.
    All DimensionalityReduction modules should return at least an 'embeddings' tensor,
    along with optional 'loss' and 'label' entries.
    """
    
    embeddings: np.ndarray
    """Reduced embeddings from the model, e.g. PHATE, UMAP, PCA."""
    
    label: NotRequired[Tensor]
    """Optional labels for the embeddings, if available."""
    
    scores: NotRequired[np.ndarray]
    """Optional scores for the embeddings, if available."""
    
    metadata: NotRequired[Any]
    """Optional metadata to be saved with the embeddings."""
    
class DimensionalityReductionCallback(BaseCallback, ABC):
    @abstractmethod
    def on_dr_end(self, dataset: Any, dr_outputs: DimensionalityReductionOutputs) -> Any:
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