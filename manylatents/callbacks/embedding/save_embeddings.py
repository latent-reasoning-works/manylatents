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
                 use_timestamp: bool = True,
                 save_additional_outputs: bool = False,
                 save_metric_tables: bool = False) -> None:
        """
        SaveEmbeddings callback that saves EmbeddingOutputs and optionally metric tables.

        Args:
            save_dir: Base directory for saving outputs (Hydra will create subdirs)
            save_format: Format for main embeddings ("csv", "npy", etc.)
            experiment_name: Name for file naming
            use_timestamp: Whether to include timestamp in names
            save_additional_outputs: Whether to save non-embeddings keys as separate files
            save_metric_tables: Whether to save separate metric table CSVs (scalar, per-sample)
        """
        super().__init__()
        self.save_dir        = save_dir
        self.save_format     = save_format
        self.experiment_name = experiment_name
        self.use_timestamp   = use_timestamp
        self.save_additional_outputs = save_additional_outputs
        self.save_metric_tables = save_metric_tables
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
        """Save embeddings as CSV."""
        df = pd.DataFrame(X, columns=[f"dim_{i+1}" for i in range(X.shape[1])])
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

    def _unpack_tuple_scores(self, raw_scores: dict) -> dict:
        """
        Unpack (scalar, per_sample_array) tuples into separate entries.
        Some metrics return both a scalar summary and per-sample values as a tuple.
        """
        scores = {}
        for name, vals in raw_scores.items():
            if isinstance(vals, tuple) and len(vals) == 2:
                scalar, arr = vals
                scores[name] = float(scalar)
                scores[f"{name}__per_sample"] = np.asarray(arr)
            else:
                scores[name] = vals
        return scores

    def _determine_n_samples(self, dataset, embeddings: dict) -> int:
        """Determine the number of samples from dataset or embeddings."""
        try:
            return len(dataset)
        except:
            # Fallback to embeddings shape
            emb = embeddings.get("embeddings")
            if emb is not None:
                emb_np = emb.numpy() if hasattr(emb, "numpy") else emb
                return emb_np.shape[0]
            return None

    def _flatten_scalar_metrics(self, scores: dict, n_samples: int) -> dict:
        """
        Flatten non-per-sample metrics into scalars.

        Returns only metrics where arr.shape[0] != n_samples (i.e., not per-sample).
        Multi-valued arrays are flattened with suffixes.
        """
        flattened = {}
        for name, value in scores.items():
            arr = np.asarray(value)

            if arr.ndim == 0:
                # Scalar
                flattened[name] = float(value)
            elif arr.shape[0] != n_samples:
                # Not per-sample: flatten array with suffixes
                if arr.ndim == 1:
                    for i, val in enumerate(arr, start=1):
                        flattened[f"{name}_{i}"] = float(val)
                else:
                    # Multi-dimensional: flatten all dimensions
                    flat_arr = arr.flatten()
                    for i, val in enumerate(flat_arr, start=1):
                        flattened[f"{name}_{i}"] = float(val)
        return flattened

    def _flatten_per_sample_metrics(self, scores: dict, n_samples: int) -> dict:
        """
        Flatten per-sample metrics (where arr.shape[0] == n_samples).

        Multi-dimensional per-sample arrays (e.g., n_samples × k) are flattened
        along non-sample dimensions with suffixes.
        """
        flattened = {}
        for name, value in scores.items():
            arr = np.asarray(value)

            if arr.ndim >= 1 and arr.shape[0] == n_samples:
                # Per-sample metric
                if arr.ndim == 1:
                    # Simple 1-D array (e.g., sample_id)
                    flattened[name] = arr
                else:
                    # Multi-dimensional (e.g., n_samples × k)
                    # Flatten along non-sample dimensions
                    if arr.ndim == 2:
                        for i in range(arr.shape[1]):
                            flattened[f"{name}_{i+1}"] = arr[:, i]
                    else:
                        # Higher dimensions: flatten to 2D then split
                        reshaped = arr.reshape(n_samples, -1)
                        for i in range(reshaped.shape[1]):
                            flattened[f"{name}_{i+1}"] = reshaped[:, i]
        return flattened

    def _save_scalar_metrics(self, scores: dict, timestamp: str) -> str:
        """Save all scalar metrics to a single-row CSV."""
        if not scores:
            logger.info("No scalar metrics to save.")
            return None

        filename = f"metrics_summary_{self.experiment_name}_{timestamp}.csv"
        save_path = os.path.join(self.save_dir, filename)

        df = pd.DataFrame([scores])
        df.to_csv(save_path, index=False)

        logger.info(f"Saved {len(scores)} scalar metrics to {save_path}")
        return save_path

    def _save_per_sample_metrics(self, scores: dict, timestamp: str) -> str:
        """Save all per-sample metrics to a CSV table (n_samples × n_metrics)."""
        if not scores:
            logger.info("No per-sample metrics to save.")
            return None

        filename = f"metrics_per_sample_{self.experiment_name}_{timestamp}.csv"
        save_path = os.path.join(self.save_dir, filename)

        df = pd.DataFrame(scores)
        df.to_csv(save_path, index=False)

        logger.info(
            f"Saved {len(scores)} per-sample metric columns with {len(df)} rows to {save_path}"
        )
        return save_path

    def on_latent_end(self, dataset: any, embeddings: dict) -> str:
        self.save_embeddings(embeddings)
        self.register_output("saved_embeddings", self.save_path)

        # Optionally save metric tables
        if self.save_metric_tables:
            raw_scores = embeddings.get("scores", {})
            if raw_scores:
                # Unpack tuple scores
                scores = self._unpack_tuple_scores(raw_scores)

                # Determine number of samples
                n_samples = self._determine_n_samples(dataset, embeddings)

                if n_samples is not None:
                    # Generate timestamp for consistent naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Flatten and save scalar metrics
                    scalar_metrics = self._flatten_scalar_metrics(scores, n_samples)
                    scalar_path = self._save_scalar_metrics(scalar_metrics, timestamp)
                    if scalar_path:
                        self.register_output("scalar_metrics_path", scalar_path)

                    # Flatten and save per-sample metrics
                    per_sample_metrics = self._flatten_per_sample_metrics(scores, n_samples)
                    per_sample_path = self._save_per_sample_metrics(per_sample_metrics, timestamp)
                    if per_sample_path:
                        self.register_output("per_sample_metrics_path", per_sample_path)
                else:
                    logger.warning("Could not determine number of samples. Skipping metric table save.")
            else:
                logger.warning("No scores found in embeddings. Skipping metric table save.")

        run = wandb.run
        if run is not None:
            rel_path = os.path.relpath(self.save_path, start=os.getcwd())
            wandb.save(rel_path, base_path=os.getcwd())
        return self.callback_outputs
