import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np
from torch import Tensor
from manylatents.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


@dataclass
class ColormapInfo:
    """
    Structured colormap information for visualization.

    This dataclass provides a standardized way for datasets to communicate
    their preferred colormap and label information to visualization callbacks.

    Attributes:
        cmap: The colormap to use. Can be:
              - A matplotlib colormap name (str), e.g., "viridis"
              - A dict mapping label values to color strings, e.g., {1: "#ff0000", 2: "#00ff00"}
              - A matplotlib ListedColormap object
        label_names: Optional mapping from numeric label indices to display names.
                    Used for legend generation with categorical data.
                    Example: {0: "Class A", 1: "Class B"}
        is_categorical: Whether the colormap represents categorical (discrete)
                       or continuous data. Affects legend vs colorbar rendering.
    """
    cmap: Union[str, Dict[Union[int, str], str], Any]  # Any for ListedColormap
    label_names: Optional[Dict[int, str]] = None
    is_categorical: bool = True


@runtime_checkable
class ColormapProvider(Protocol):
    """
    Protocol for datasets that can provide colormap information.

    Datasets implementing this protocol can specify their preferred colormap
    and label names, allowing PlotEmbeddings to render them appropriately
    without requiring isinstance checks for specific dataset types.

    Example:
        class MyDataset(ColormapProvider):
            def get_colormap_info(self) -> ColormapInfo:
                return ColormapInfo(
                    cmap={"A": "#ff0000", "B": "#00ff00"},
                    label_names={0: "Category A", 1: "Category B"},
                    is_categorical=True
                )
    """

    def get_colormap_info(self) -> ColormapInfo:
        """
        Return colormap information for visualization.

        Returns:
            ColormapInfo with cmap, optional label_names, and is_categorical flag.
        """
        ...

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