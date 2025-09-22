import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from torch import Tensor
from manylatents.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

EmbeddingOutputs = dict[str, Any]
"""
Standard data interchange format for manyLatents pipeline.

Required:
    embeddings (np.ndarray): Primary reduced dimensionality output

Standard optional keys:
    label: Labels/targets for data samples
    metadata: Algorithm parameters and info
    scores: Evaluation metrics

Custom keys are encouraged for algorithm-specific outputs.
"""

def validate_embedding_outputs(outputs: EmbeddingOutputs) -> EmbeddingOutputs:
    """
    Validates that EmbeddingOutputs contains the required 'embeddings' key.

    Args:
        outputs: Dictionary containing embedding outputs

    Returns:
        The validated outputs dictionary

    Raises:
        ValueError: If not a dictionary or missing 'embeddings' key
    """
    if not isinstance(outputs, dict):
        raise ValueError("EmbeddingOutputs must be a dictionary")

    if "embeddings" not in outputs:
        raise ValueError("EmbeddingOutputs must contain an 'embeddings' key")

    return outputs
    
class EmbeddingCallback(BaseCallback, ABC):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_outputs = {}

    def register_output(self, key: str, output: Any) -> None:
        """Store a callback output for later use."""
        self.callback_outputs[key] = output
        logger.info(f"Output registered under key: {key}")

    @abstractmethod
    def on_latent_end(self, dataset: Any, embeddings: EmbeddingOutputs) -> Any:
        """
        Called when the latent process is complete.
        
        Args:
            dataset (Any): A dataset object that exposes properties/methods 
                           like `full_data` and `get_labels()`.
            embeddings (np.ndarray): The computed embeddings.
        
        Returns:
            Any: A result, for example a file path or a dictionary of metrics.
        """
        pass