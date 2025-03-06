import logging
import os
import warnings
from datetime import datetime

import numpy as np

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback
from src.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(DimensionalityReductionCallback):
    def __init__(self, save_dir: str = "outputs", save_format: str = "npy") -> None:
        self.save_dir = save_dir
        self.save_format = save_format
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.debug(f"save_embeddings() called with embeddings shape: {embeddings.shape}")
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
        logger.debug("on_dr_end() called; delegating to save_embeddings()")
        self.save_embeddings(embeddings, labels)


