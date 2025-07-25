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

    def on_latent_end(self, dataset, embeddings: dict) -> dict:
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

        # 2) per-sample table (1-D arrays) - handle variable lengths
        array_keys = [k for k, v in scores.items() if np.ndim(v) == 1]
        if self.log_table and array_keys:
            # Find the maximum length among all arrays
            lengths = [len(scores[k]) for k in array_keys]
            max_length = max(lengths)
            min_length = min(lengths)
            
            if max_length != min_length:
                logger.warning(f"Arrays have different lengths: {dict(zip(array_keys, lengths))}. "
                             f"Using max length {max_length} and padding shorter arrays with NaN.")
            
            data = {"sample_index": np.arange(max_length)}
            
            if self.include_labels and "label" in embeddings:
                lbl = embeddings["label"]
                lbl_list = (
                    lbl.cpu().numpy().tolist() if hasattr(lbl, "cpu")
                    else list(lbl)
                )
                # Pad labels if needed
                if len(lbl_list) < max_length:
                    lbl_list.extend([None] * (max_length - len(lbl_list)))
                elif len(lbl_list) > max_length:
                    lbl_list = lbl_list[:max_length]
                data["label"] = lbl_list
            
            for name in array_keys:
                arr = scores[name]
                arr_list = arr.tolist()
                
                # Pad shorter arrays with NaN
                if len(arr_list) < max_length:
                    arr_list.extend([np.nan] * (max_length - len(arr_list)))
                elif len(arr_list) > max_length:
                    arr_list = arr_list[:max_length]
                
                data[name] = arr_list

            df = pd.DataFrame(data)
            table = wandb.Table(dataframe=df)
            wandb.log({f"{tag}/per_sample_metrics": table}, commit=False)
            logger.info(f"Logged table {tag}/per_sample_metrics with {len(df)} rows")

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