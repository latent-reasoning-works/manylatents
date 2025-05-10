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

        scores = embeddings.get("scores", {})
        if not scores:
            logger.warning("No scores found.")
            return {}

        # figure out whether this is DR or latent
        tag = embeddings.get("metadata", {}).get("source", "embedding")

        # --- 1) summary scalars (0-D only) ---
        scalar_summary = {
            f"{tag}/{name}": float(vals)
            for name, vals in scores.items()
            if np.ndim(vals) == 0
        }
        if self.log_summary and scalar_summary:
            wandb.log(scalar_summary, commit=False)
            logger.info(f"Logged scalars for {tag}: {list(scalar_summary)}")

        # --- 2) per-sample table (1-D arrays) ---
        array_keys = [k for k,v in scores.items() if np.ndim(v) > 0]
        if self.log_table and array_keys:
            N = len(scores[array_keys[0]])
            data = {"sample_index": np.arange(N)}
            if self.include_labels and "label" in embeddings:
                lbl = embeddings["label"]
                data["label"] = (
                    lbl.cpu().numpy().tolist() if hasattr(lbl, "cpu")
                    else list(lbl)
                )
            for name in array_keys:
                data[name] = np.asarray(scores[name]).tolist()

            df = pd.DataFrame(data)
            table = wandb.Table(dataframe=df)
            wandb.log({f"{tag}/per_sample_metrics": table}, commit=False)
            logger.info(f"Logged table {tag}/per_sample_metrics")

        # --- 3) k-curve tables ---
        if self.log_k_curve_table and scalar_summary:
            # group all swept scalars by their base name
            groups: dict[str,list[tuple[int,float]]] = {}
            for full_name, val in scalar_summary.items():
                # strip off the tag/ prefix
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
