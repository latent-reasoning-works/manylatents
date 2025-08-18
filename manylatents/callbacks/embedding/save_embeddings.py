import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import wandb
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(EmbeddingCallback):
    def __init__(self, 
                 save_dir: str = "outputs", 
                 save_format: str = "npy", 
                 experiment_name: str = "experiment",
                 include_metrics: bool = False,
                 use_timestamp: bool = True) -> None:
        super().__init__()
        self.save_dir        = save_dir
        self.save_format     = save_format
        self.experiment_name = experiment_name
        self.include_metrics = include_metrics
        self.use_timestamp   = use_timestamp
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: dict) -> None:
        X = embeddings["embeddings"]
        metadata = embeddings.get("metadata", {}) or {}
        if "labels" not in metadata and "label" in embeddings:
            metadata["labels"] = embeddings["label"]

        if self.use_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"embeddings_{self.experiment_name}_{ts}.{self.save_format}"
        else:
            fname = f"embeddings_{self.experiment_name}.{self.save_format}"
        self.save_path = os.path.join(self.save_dir, fname)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        logger.debug(f"save_embeddings() called with embeddings shape: {X.shape}")
        logger.info(f"Computed save path: {self.save_path}")

        if self.save_format.lower() == "csv":
            logger.info("Saving embeddings as CSV.")
            df = pd.DataFrame(
                data=X,
                columns=[f"dim_{i+1}" for i in range(X.shape[1])]
            )

            if self.include_metrics and "scores" in embeddings:
                scores = embeddings["scores"]
                if isinstance(scores, dict):
                    for key, val in scores.items():
                        arr = np.asarray(val)
                        # only write 1D arrays that match #samples
                        if arr.ndim == 1 and arr.shape[0] == df.shape[0]:
                            df[key] = arr
                        else:
                            logger.debug(f"Skipping CSV column for '{key}'; shape={arr.shape}")
                else:
                    # fallback if someone passed a bare array
                    arr = np.asarray(scores)
                    if arr.ndim == 1 and arr.shape[0] == df.shape[0]:
                        df["scores"] = arr
                    else:
                        logger.debug("Skipping CSV column for 'scores'; not a 1D array")

            df.to_csv(self.save_path, index=False)

        else:
            # for .npy, .pt, etc.
            save_embeddings(X, self.save_path, format=self.save_format, metadata=metadata)

        logger.info(f"Saved embeddings successfully to {self.save_path}")

    def on_latent_end(self, dataset: any, embeddings: dict) -> str:
        self.save_embeddings(embeddings)
        self.register_output("saved_embeddings", self.save_path)

        run = wandb.run
        if run is not None:
            rel_path = os.path.relpath(self.save_path, start=os.getcwd())
            wandb.save(rel_path, base_path=os.getcwd())
        return self.callback_outputs
