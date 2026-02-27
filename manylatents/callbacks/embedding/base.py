import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np
from torch import Tensor
from manylatents.callbacks.callback import BaseCallback

logger = logging.getLogger(__name__)


@dataclass
class ColormapInfo:
    """
    Structured colormap information for visualization.

    This dataclass provides a standardized way for datasets and metrics to communicate
    their preferred colormap and label information to visualization callbacks.

    Attributes:
        cmap: The colormap to use. Can be:
              - A matplotlib colormap name (str), e.g., "viridis"
              - A dict mapping label values to color strings, e.g., {1: "#ff0000", 2: "#00ff00"}
              - A matplotlib ListedColormap object
              - Special values: "categorical" (auto-generate discrete colors)
        label_names: Optional mapping from label values to display names.
                    Used for legend generation with categorical data.
                    Example: {0: "Class A", 1: "Class B"}
                    If None and label_format is set, labels are generated dynamically.
        label_format: Optional format string for dynamic label generation.
                     Example: "Dim = {}" generates "Dim = 1", "Dim = 2", etc.
                     Used when label_names is None and is_categorical is True.
        is_categorical: Whether the colormap represents categorical (discrete)
                       or continuous data. Affects legend vs colorbar rendering.
    """
    cmap: Union[str, Dict[Union[int, str], str], Any]  # Any for ListedColormap
    label_names: Optional[Dict[Union[int, str], str]] = None
    label_format: Optional[str] = None
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

LatentOutputs = dict[str, Any]
"""
Standard data interchange format for the manylatents pipeline.

Required:
    embeddings (np.ndarray): Primary reduced dimensionality output, shape (n, d)

Standard optional keys:
    label: Labels/targets for data samples
    metadata: Algorithm parameters and info
    scores: Evaluation metrics

Custom keys are encouraged for algorithm-specific outputs
(e.g., trajectories, cluster assignments, velocity fields).
"""

# Backward-compatible alias (deprecated)
EmbeddingOutputs = LatentOutputs


def validate_latent_outputs(outputs: LatentOutputs) -> LatentOutputs:
    """
    Validates that LatentOutputs contains the required 'embeddings' key.

    Args:
        outputs: Dictionary containing latent outputs

    Returns:
        The validated outputs dictionary

    Raises:
        ValueError: If not a dictionary or missing 'embeddings' key
    """
    if not isinstance(outputs, dict):
        raise ValueError("LatentOutputs must be a dictionary")

    if "embeddings" not in outputs:
        raise ValueError("LatentOutputs must contain an 'embeddings' key")

    return outputs


# Backward-compatible alias (deprecated)
validate_embedding_outputs = validate_latent_outputs
    
class EmbeddingCallback(BaseCallback, ABC):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_outputs = {}

    def register_output(self, key: str, output: Any) -> None:
        """Store a callback output for later use."""
        self.callback_outputs[key] = output
        logger.info(f"Output registered under key: {key}")

    @abstractmethod
    def on_latent_end(self, dataset: Any, embeddings: LatentOutputs) -> Any:
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