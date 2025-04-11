import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from torch import Tensor
from typing_extensions import NotRequired, TypedDict

from src.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

class EmbeddingOutputs(TypedDict, total=False):
    """
    Defines the minimal output contract for embedding methods.
    All embedding modules (DR or NN encoders) should return at least an 'embeddings' array,
    along with optional 'label', 'scores', and 'metadata' entries.
    """
    embeddings: np.ndarray
    """Reduced embeddings from the model, e.g. UMAP, PCA, or an encoder output."""
    
    label: NotRequired[Tensor]
    """Optional labels for the embeddings, if available."""
    
    scores: NotRequired[np.ndarray]
    """Optional scores for the embeddings, if available (e.g. tangent_space values)."""
    
    metadata: NotRequired[Any]
    """Optional metadata to be saved with the embeddings."""
    
class EmbeddingCallback(BaseCallback, ABC):
    @abstractmethod
    def on_dr_end(self, dataset: Any, embeddings: EmbeddingOutputs) -> Any:
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