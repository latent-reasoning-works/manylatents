import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.callbacks.embedding.base import EmbeddingCallback
from src.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(EmbeddingCallback):
    def __init__(self, 
                 save_dir: str = "outputs", 
                 save_format: str = "npy", 
                 experiment_name: str = "experiment",
                 include_metrics: bool=False) -> None:
        
        self.save_dir = save_dir
        self.save_format = save_format
        self.experiment_name = experiment_name
        self.include_metrics = include_metrics
        
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: dict) -> None:
        _embeddings = embeddings["embeddings"]
        metadata = embeddings.get("metadata")
        if metadata is None:
            metadata = {}
        if "labels" not in metadata and "label" in embeddings:
            metadata["labels"] = embeddings["label"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embeddings_{self.experiment_name}_{timestamp}.{self.save_format}"
        self.save_path = os.path.join(self.save_dir, filename)
        logger.debug(f"save_embeddings() called with embeddings shape: {_embeddings.shape}")
        logger.info(f"Computed save path: {self.save_path}")

        # If saving as CSV and include_metrics is True, combine embeddings and scores.
        if self.save_format.lower() == "csv":
            logger.info("Saving embeddings as CSV.")
            df = pd.DataFrame(_embeddings, columns=[f"dim_{i+1}" for i in range(_embeddings.shape[1])])
            if self.include_metrics and "scores" in embeddings:
                scores = embeddings["scores"]
                # If scores is a dict, iterate over keys and add each as a column.
                if isinstance(scores, dict):
                    for key, value in scores.items():
                        # Only add if the array length matches the number of samples.
                        if isinstance(value, np.ndarray) and value.shape[0] == df.shape[0]:
                            df[key] = value
                        else:
                            logger.warning(f"Metric '{key}' could not be added to CSV (shape mismatch or wrong type).")
                else:
                    df["scores"] = scores
            df.to_csv(self.save_path, index=False)
        else:
            save_embeddings(_embeddings, self.save_path, format=self.save_format, metadata=metadata)
        logger.info(f"Saved embeddings successfully to {self.save_path}")

    def on_dr_end(self, dataset: any, embeddings: dict) -> str:
        logger.debug("on_dr_end() called; delegating to save_embeddings()")
        # If labels weren't provided in embeddings, extract them from the dataset.
        # note: may be a redundant check, given how they're accessed in main
        if "label" not in embeddings and hasattr(dataset, "get_labels"):
            embeddings["label"] = dataset.get_labels()
        self.save_embeddings(embeddings)
        return self.save_path

