import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import numpy as np
import torch
from torch import Tensor
from typing_extensions import NotRequired, Required, TypedDict

from src.utils.utils import save_embeddings

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
    
class DimensionalityReductionCallback(ABC):
    @abstractmethod
    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Called when the Dimensionality Reduction process is complete.
        
        Args:
            embeddings (np.ndarray): The computed embeddings.
            labels (np.ndarray, optional): Optional labels associated with the embeddings.
        """
        pass

class SaveEmbeddings(DimensionalityReductionCallback):
    def __init__(self, save_dir: str = "outputs", save_format: str = "npy") -> None:
        self.save_dir = save_dir
        self.save_format = save_format
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.info(f"save_embeddings() called with embeddings shape: {embeddings.shape}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{timestamp}.{self.save_format}"
        save_path = os.path.join(self.save_dir, filename)
        logger.info(f"Computed save path: {save_path}")
        
        metadata = {"labels": labels} if labels is not None else None

        try:
            save_embeddings(embeddings, save_path, format=self.save_format, metadata=metadata)
            logger.info(f"Saved embeddings successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            warnings.warn(f"Failed to save embeddings: {e}")

    def on_dr_end(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.info("on_dr_end() called; delegating to save_embeddings()")
        self.save_embeddings(embeddings, labels)
