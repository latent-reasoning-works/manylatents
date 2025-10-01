import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from manylatents.callbacks.embedding.base import EmbeddingCallback

logger = logging.getLogger(__name__)


class SaveMetricTables(EmbeddingCallback):
    """
    Callback that saves metric information to CSV files for offline analysis.

    Saves three types of outputs:
    1. Scalar metrics (0-D) → single-row CSV with all scalar values
    2. Per-sample metrics (1-D arrays) → table with sample_index and metric columns
    3. K-curve metrics (swept scalars) → separate CSV per metric base name
    """

    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        include_sample_index: bool = True,
        include_labels: bool = True,
    ):
        """
        Args:
            save_dir: Directory where CSV files will be saved.
            experiment_name: Name of the experiment for filenames.
            include_sample_index: Whether to include sample_index column in per-sample table.
            include_labels: Whether to include label column in per-sample table if available.
        """
        super().__init__()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.include_sample_index = include_sample_index
        self.include_labels = include_labels

        # Regex to match k-sweep patterns like "foo__n_neighbors_15"
        self._knn_re = re.compile(r"(?P<base>.+)__n_neighbors_(?P<k>\d+)$")

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"SaveMetricTables initialized with directory: {self.save_dir} "
            f"and experiment name: {self.experiment_name}"
        )

    def _unpack_tuple_scores(self, raw_scores: dict) -> dict:
        """
        Unpack (scalar, per_sample_array) tuples into separate entries.
        Follows the same logic as WandbLogScores.
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

    def _save_scalar_metrics(self, scores: dict, timestamp: str) -> str:
        """Save all 0-D scalar metrics to a single-row CSV."""
        scalar_metrics = {
            name: float(v)
            for name, v in scores.items()
            if np.ndim(v) == 0
        }

        if not scalar_metrics:
            logger.info("No scalar metrics to save.")
            return None

        filename = f"metrics_summary_{self.experiment_name}_{timestamp}.csv"
        save_path = os.path.join(self.save_dir, filename)

        df = pd.DataFrame([scalar_metrics])
        df.to_csv(save_path, index=False)

        logger.info(f"Saved {len(scalar_metrics)} scalar metrics to {save_path}")
        return save_path

    def _save_per_sample_metrics(
        self,
        scores: dict,
        embeddings: dict,
        dataset,
        timestamp: str
    ) -> str:
        """Save all 1-D per-sample metrics to a CSV table."""
        array_keys = [k for k, v in scores.items() if np.ndim(v) == 1]

        if not array_keys:
            logger.info("No per-sample metrics to save.")
            return None

        # Determine canonical length from dataset (for padding if needed)
        try:
            canonical_length = len(dataset)
        except Exception as e:
            logger.warning(f"Could not get dataset length: {e}. Using metric array length.")
            canonical_length = None

        # Find maximum length from metric arrays
        metric_lengths = [len(scores[k]) for k in array_keys]
        max_metric_length = max(metric_lengths)
        min_metric_length = min(metric_lengths)

        # Determine the final length for the table
        if canonical_length is not None and max_metric_length != canonical_length:
            logger.warning(
                f"Metric arrays have length {max_metric_length} but dataset has {canonical_length} samples. "
                f"Using dataset length {canonical_length} and padding metric arrays with NaN."
            )
            max_length = canonical_length
        else:
            max_length = max_metric_length

        if min_metric_length != max_metric_length:
            logger.warning(
                f"Metric arrays have different lengths: {dict(zip(array_keys, metric_lengths))}. "
                f"Padding shorter arrays with NaN to match length {max_length}."
            )

        data = {}

        # Add labels if available
        if self.include_labels and "label" in embeddings:
            lbl = embeddings["label"]
            lbl_list = (
                lbl.cpu().numpy().tolist() if hasattr(lbl, "cpu")
                else list(lbl)
            )
            # Pad or truncate to match max_length
            if len(lbl_list) < max_length:
                lbl_list.extend([None] * (max_length - len(lbl_list)))
            elif len(lbl_list) > max_length:
                lbl_list = lbl_list[:max_length]
            data["label"] = lbl_list

        # Add all per-sample metric arrays
        for name in array_keys:
            arr = scores[name]
            # Handle pandas Series or numpy arrays
            if hasattr(arr, 'values'):
                arr_list = arr.values.tolist()
            else:
                arr_list = arr.tolist()

            # Pad or truncate to match max_length
            if len(arr_list) < max_length:
                arr_list.extend([np.nan] * (max_length - len(arr_list)))
            elif len(arr_list) > max_length:
                arr_list = arr_list[:max_length]

            data[name] = arr_list

        filename = f"metrics_per_sample_{self.experiment_name}_{timestamp}.csv"
        save_path = os.path.join(self.save_dir, filename)

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

        logger.info(
            f"Saved {len(array_keys)} per-sample metrics with {len(df)} rows to {save_path}"
        )
        return save_path

    def _save_k_curve_metrics(self, scores: dict, timestamp: str) -> list:
        """
        Save k-curve metrics (swept scalars with __n_neighbors_N pattern).
        Creates separate CSV files for each base metric name.
        """
        # Extract only scalar metrics for k-curve analysis
        scalar_metrics = {
            name: float(v)
            for name, v in scores.items()
            if np.ndim(v) == 0
        }

        # Group metrics by base name
        groups = {}
        for name, val in scalar_metrics.items():
            m = self._knn_re.match(name)
            if m:
                base = m.group("base")
                k = int(m.group("k"))
                groups.setdefault(base, []).append((k, val))

        if not groups:
            logger.info("No k-curve metrics to save.")
            return []

        saved_paths = []
        for base, kv_pairs in groups.items():
            if len(kv_pairs) < 2:
                logger.debug(f"Skipping k-curve for '{base}' (only 1 data point)")
                continue

            # Sort by k value
            kv_pairs.sort(key=lambda x: x[0])
            ks, vs = zip(*kv_pairs)

            filename = f"metrics_k_curve_{base}_{self.experiment_name}_{timestamp}.csv"
            save_path = os.path.join(self.save_dir, filename)

            df = pd.DataFrame({"n_neighbors": ks, base: vs})
            df.to_csv(save_path, index=False)

            logger.info(f"Saved k-curve for '{base}' to {save_path}")
            saved_paths.append(save_path)

        return saved_paths

    def on_latent_end(self, dataset, embeddings: dict) -> dict:
        """
        Main callback method called when latent processing completes.
        Extracts and saves all metric tables.
        """
        raw_scores = embeddings.get("scores", {})
        if not raw_scores:
            logger.warning("No scores found in embeddings. Skipping metric table save.")
            return self.callback_outputs

        # Unpack tuple scores (scalar, per_sample_array)
        scores = self._unpack_tuple_scores(raw_scores)

        # Generate timestamp for consistent naming across files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save all three types of metrics
        scalar_path = self._save_scalar_metrics(scores, timestamp)
        per_sample_path = self._save_per_sample_metrics(scores, embeddings, dataset, timestamp)
        k_curve_paths = self._save_k_curve_metrics(scores, timestamp)

        # Register outputs for potential downstream use
        if scalar_path:
            self.register_output("scalar_metrics_path", scalar_path)
        if per_sample_path:
            self.register_output("per_sample_metrics_path", per_sample_path)
        if k_curve_paths:
            self.register_output("k_curve_metrics_paths", k_curve_paths)

        logger.info("SaveMetricTables completed successfully.")
        return self.callback_outputs
