import logging
import os
from datetime import datetime

import numpy as np

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback
from src.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(DimensionalityReductionCallback):
    def __init__(self, save_dir: str = "outputs", 
                 save_format: str = "npy", 
                 experiment_name: str = "experiment") -> None:
        self.save_dir = save_dir
        self.save_format = save_format
        self.experiment_name = experiment_name
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        logger.debug(f"save_embeddings() called with embeddings shape: {embeddings.shape}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{self.experiment_name}_{timestamp}.{self.save_format}"
        self.save_path = os.path.join(self.save_dir, filename)
        logger.info(f"Computed save path: {self.save_path}")
        
        metadata = {"labels": labels} if labels is not None else None

        try:
            save_embeddings(embeddings, self.save_path, format=self.save_format, metadata=metadata)
            logger.info(f"Saved embeddings successfully to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
 
    def on_dr_end(self, dataset: any, embeddings: np.ndarray) -> str:
        logger.debug("on_dr_end() called; delegating to save_embeddings()")
        # Try to extract labels from the dataset if available.
        try:
            labels = dataset.get_labels()
        except Exception as e:
            logger.debug(f"No labels available from dataset: {e}")
            labels = None
        self.save_embeddings(embeddings, labels)
        return self.save_path
