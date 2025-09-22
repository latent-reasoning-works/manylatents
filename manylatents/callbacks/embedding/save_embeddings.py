import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import wandb
from manylatents.callbacks.embedding.base import EmbeddingCallback, validate_embedding_outputs
from manylatents.utils.utils import save_embeddings

logger = logging.getLogger(__name__)

class SaveEmbeddings(EmbeddingCallback):
    def __init__(self,
                 save_dir: str = "outputs",
                 save_format: str = "npy",
                 experiment_name: str = "experiment",
                 include_metrics: bool = False,
                 use_timestamp: bool = True,
                 save_additional_outputs: bool = False) -> None:
        """
        SaveEmbeddings callback that saves EmbeddingOutputs.

        Args:
            save_dir: Base directory for saving outputs (Hydra will create subdirs)
            save_format: Format for main embeddings ("csv", "npy", etc.)
            experiment_name: Name for file naming
            include_metrics: Whether to include scores/metrics in CSV format
            use_timestamp: Whether to include timestamp in names
            save_additional_outputs: Whether to save non-embeddings keys as separate files
        """
        super().__init__()
        self.save_dir        = save_dir
        self.save_format     = save_format
        self.experiment_name = experiment_name
        self.include_metrics = include_metrics
        self.use_timestamp   = use_timestamp
        self.save_additional_outputs = save_additional_outputs
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"SaveEmbeddings initialized with directory: {self.save_dir} and format: {self.save_format}")

    def save_embeddings(self, embeddings: dict) -> None:
        """Save EmbeddingOutputs - main embeddings + optionally additional outputs."""
        embeddings = validate_embedding_outputs(embeddings)

        base_name = self._get_base_filename()
        self._save_main_embeddings(embeddings, base_name)

        if self.save_additional_outputs:
            self._save_additional_outputs(embeddings, base_name)

    def _get_base_filename(self) -> str:
        """Generate base filename with optional timestamp."""
        if self.use_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"embeddings_{self.experiment_name}_{ts}"
        else:
            return f"embeddings_{self.experiment_name}"

    def _get_save_path(self, filename: str) -> str:
        """Generate full save path and ensure directory exists."""
        path = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _to_numpy(self, value):
        """Convert tensor to numpy if needed."""
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    def _save_main_embeddings(self, embeddings: dict, base_name: str) -> None:
        """Save the main embeddings in the specified format."""
        X = self._to_numpy(embeddings["embeddings"])
        filename = f"{base_name}.{self.save_format}"
        self.save_path = self._get_save_path(filename)

        logger.info(f"Saving embeddings with shape {X.shape} to {self.save_path}")

        if self.save_format.lower() == "csv":
            self._save_csv(embeddings, X)
        else:
            self._save_single_file(embeddings, X)

    def _save_csv(self, embeddings: dict, X: np.ndarray) -> None:
        """Save as CSV with optional additional columns."""
        df = pd.DataFrame(X, columns=[f"dim_{i+1}" for i in range(X.shape[1])])

        if self.include_metrics:
            for key, value in embeddings.items():
                if key == "embeddings":
                    continue
                value = self._to_numpy(value)
                if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(df):
                    df[key] = value
                elif isinstance(value, (list, tuple)) and len(value) == len(df):
                    df[key] = value

        df.to_csv(self.save_path, index=False)

    def _save_single_file(self, embeddings: dict, X: np.ndarray) -> None:
        """Save using existing utility function."""
        metadata = embeddings.get("metadata", {})
        if "label" in embeddings and "labels" not in metadata:
            metadata["labels"] = embeddings["label"]
        save_embeddings(X, self.save_path, format=self.save_format, metadata=metadata)

    def _save_additional_outputs(self, embeddings: dict, base_name: str) -> None:
        """Save additional outputs as separate files."""
        for key, value in embeddings.items():
            if key in ["embeddings", "metadata"]:
                continue

            value = self._to_numpy(value)

            if isinstance(value, np.ndarray):
                filename = f"{base_name}_{key}.npy"
                path = self._get_save_path(filename)
                np.save(path, value)
                logger.info(f"Saved {key} to {path}")
            elif isinstance(value, (dict, list)):
                filename = f"{base_name}_{key}.json"
                path = self._get_save_path(filename)
                with open(path, 'w') as f:
                    json.dump(value, f, indent=2, default=str)
                logger.info(f"Saved {key} to {path}")

    def on_latent_end(self, dataset: any, embeddings: dict) -> str:
        self.save_embeddings(embeddings)
        self.register_output("saved_embeddings", self.save_path)

        run = wandb.run
        if run is not None:
            rel_path = os.path.relpath(self.save_path, start=os.getcwd())
            wandb.save(rel_path, base_path=os.getcwd())
        return self.callback_outputs
