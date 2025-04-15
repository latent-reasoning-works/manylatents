import logging
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch

import wandb
from src.callbacks.embedding.base import EmbeddingCallback

logger = logging.getLogger(__name__)

class WandbLogEmbeddings(EmbeddingCallback):
    def __init__(
        self,
        log_embeddings: bool = True,
        log_metrics: bool = True,
        include_labels: bool = True,
        max_dims_to_log: int = 10,
        log_archetype_similarity: bool = False,
    ):
        super().__init__()  
        self.log_embeddings = log_embeddings
        self.log_metrics = log_metrics
        self.include_labels = include_labels
        self.max_dims_to_log = max_dims_to_log
        self.log_archetype_similarity = log_archetype_similarity

    def on_dr_end(
        self,
        embeddings: Dict[str, Any],
        dataset: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        if wandb.run is None:
            logger.info("W&B logging skipped: wandb.run is None")
            return {}

        wandb.run.summary["cfg_name"] = getattr(wandb.config, "name", "unknown")
        wandb.run.summary["slurm_id"] = os.environ.get("SLURM_JOB_ID", "n/a")
        wandb.run.summary["slurm_node"] = os.environ.get("SLURMD_NODENAME", "n/a")

        out = {}
        emb_tensor = embeddings.get("embeddings")
        if isinstance(emb_tensor, torch.Tensor):
            emb_array = emb_tensor.detach().cpu().numpy()
        else:
            emb_array = emb_tensor

        labels = embeddings.get("label") if self.include_labels else None
        metadata = embeddings.get("metadata", {})

        # 1. Log embeddings as W&B Table
        if self.log_embeddings:
            num_samples, num_dims = emb_array.shape
            dims_to_log = min(num_dims, self.max_dims_to_log)
            columns = ["sample_id"] + (["label"] if labels is not None else []) + [f"dim_{i}" for i in range(dims_to_log)]
            table_data = []
            for i in range(num_samples):
                row = [i]
                if labels is not None:
                    row.append(labels[i])
                row.extend(emb_array[i, :dims_to_log])
                table_data.append(row)
            embedding_table = wandb.Table(data=table_data, columns=columns)
            key = f"embedding_table_{wandb.run.name}"
            wandb.log({key: embedding_table})
            out["embedding_table"] = embedding_table
            logger.info("Logged embedding_table to W&B")
            # Register in the output registry
            self.register_output(key, embedding_table)

        # 2. Log per-sample metrics as W&B Table
        if self.log_metrics and "scores" in embeddings:
            scores = embeddings["scores"]
            metric_keys = [k for k, v in scores.items() if hasattr(v, '__len__') and not isinstance(v, str)]
            if metric_keys:
                num_samples = len(scores[metric_keys[0]])
                columns = ["sample_id"] + (["label"] if labels is not None else []) + metric_keys
                table_data = []
                for i in range(num_samples):
                    row = [i]
                    if labels is not None:
                        row.append(labels[i])
                    row.extend([scores[k][i] for k in metric_keys])
                    table_data.append(row)
                metrics_table = wandb.Table(data=table_data, columns=columns)
                key = f"metrics_table_{wandb.run.name}"
                wandb.log({key: metrics_table})
                out["metrics_table"] = metrics_table
                logger.info("Logged metrics_table to W&B")
                self.register_output(key, metrics_table)

        # 3. Log archetype similarity if available
        if self.log_archetype_similarity and "archetype_similarity" in embeddings:
            similarity = embeddings["archetype_similarity"]
            if isinstance(similarity, torch.Tensor):
                similarity = similarity.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(12, 3))
            from archetypes.visualization import stacked_bar
            stacked_bar(similarity, ax=ax)
            plt.title("Archetype Similarity (Soft Assignments)")
            key = "archetype_similarity_barplot"
            wandb.log({key: wandb.Image(fig)})
            plt.close(fig)
            logger.info("Logged archetype similarity barplot to W&B")
            self.register_output(key, wandb.Image(fig))
        
        # Optionally, return the entire callback output registry.
        return self.callback_outputs
