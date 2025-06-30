import logging
import re

import numpy as np
import pandas as pd

import wandb
from src.callbacks.embedding.base import EmbeddingCallback

logger = logging.getLogger(__name__)

class WandbLogScores(EmbeddingCallback):
    """
    1) 0-D scalars → one wandb.log() with keys like "DR/…"
    2) per-sample table → key "DR/per_sample_metrics"
    3) k-curve tables → keys like "DR/continuity__k_curve_table"
    """
    def __init__(
        self,
        log_summary:         bool = True,
        log_table:           bool = True,
        log_k_curve_table:   bool = True,
        include_labels:      bool = False,
    ):
        super().__init__()
        self.log_summary       = log_summary
        self.log_table         = log_table
        self.log_k_curve_table = log_k_curve_table
        self.include_labels    = include_labels

        # matches "foo__n_neighbors_15" etc.
        self._knn_re = re.compile(r"(?P<base>.+)__n_neighbors_(?P<k>\d+)$")

    def on_dr_end(self, dataset, embeddings: dict) -> dict:
        run = wandb.run
        if run is None:
            return {}

        raw_scores = embeddings.get("scores", {})
        if not raw_scores:
            logger.warning("No scores found.")
            return {}

        # 0) first, unpack any (scalar, per_sample_array) tuples into two entries
        scores: dict[str, np.ndarray | float] = {}
        for name, vals in raw_scores.items():
            if isinstance(vals, tuple) and len(vals) == 2:
                scalar, arr = vals
                scores[name] = float(scalar)
                scores[f"{name}__per_sample"] = np.asarray(arr)
            else:
                scores[name] = vals

        tag = embeddings.get("metadata", {}).get("source", "embedding")

        # 1) summary scalars (0-D only)
        scalar_summary = {
            f"{tag}/{name}": float(v)
            for name, v in scores.items()
            if np.ndim(v) == 0
        }
        if self.log_summary and scalar_summary:
            wandb.log(scalar_summary, commit=False)
            logger.info(f"Logged scalars for {tag}: {list(scalar_summary)}")

        # 2) per-sample table (1-D arrays)
        array_keys = [k for k, v in scores.items() if np.ndim(v) == 1]
        if self.log_table and array_keys:
            # Group arrays by their length
            length_to_keys = {}
            for name in array_keys:
                arr_len = len(scores[name])
                length_to_keys.setdefault(arr_len, []).append(name)

            for N, keys in length_to_keys.items():
                data = {"sample_index": np.arange(N)}
                # Only add label if it matches this group length
                if self.include_labels and "label" in embeddings:
                    lbl = embeddings["label"]
                    if hasattr(lbl, "cpu"):
                        lbl = lbl.cpu().numpy()
                    lbl = np.asarray(lbl)
                    if len(lbl) == N:
                        data["label"] = lbl.tolist()
                    else:
                        logger.warning(f"Label length {len(lbl)} does not match group length {N}; skipping label for this table.")
                for name in keys:
                    arr = scores[name]
                    if len(arr) == N:
                        data[name] = arr.tolist()
                    else:
                        logger.warning(f"Array '{name}' length {len(arr)} does not match group length {N}; skipping.")
                try:
                    df = pd.DataFrame(data)
                    table_key = f"{tag}/per_sample_metrics_n{N}" if len(length_to_keys) > 1 else f"{tag}/per_sample_metrics"
                    table = wandb.Table(dataframe=df)
                    wandb.log({table_key: table}, commit=False)
                    logger.info(f"Logged table {table_key} with columns: {list(data.keys())}")
                except Exception as e:
                    logger.error(f"Failed to log per-sample table for group length {N}: {e}")
            if len(length_to_keys) > 1:
                logger.warning(f"Multiple per-sample array groups found (lengths: {list(length_to_keys.keys())}); logged separately.")

        # 3) k-curve tables (group swept scalars by base name)
        if self.log_k_curve_table and scalar_summary:
            groups: dict[str, list[tuple[int, float]]] = {}
            for full_name, val in scalar_summary.items():
                _, inner = full_name.split("/", 1)
                m = self._knn_re.match(inner)
                if m:
                    base = m.group("base")
                    k    = int(m.group("k"))
                    groups.setdefault(base, []).append((k, val))

            for base, kv in groups.items():
                if len(kv) < 2:
                    continue
                kv.sort(key=lambda x: x[0])
                ks, vs = zip(*kv)
                df = pd.DataFrame({"n_neighbors": ks, base: vs})
                table = wandb.Table(dataframe=df)
                wandb.log({f"{tag}/{base}__k_curve_table": table}, commit=False)
                logger.info(f"Logged {tag}/{base}__k_curve_table: ks={ks}, vals={vs}")

        # final flush
        wandb.log({})
        return {}