import logging

import numpy as np

import wandb
from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback

logger = logging.getLogger(__name__)

def safe_wandb_log(metrics_dict):
    # Only log if wandb.run is available (not in debug mode)
    # placeholder
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.log(metrics_dict)
    else:
        logger.info("wandb.run not active; skipping wandb.log")
        
class AdditionalMetrics(DimensionalityReductionCallback):
    def __init__(self, metadata=None, **kwargs):
        """
        Store metric configs as they are (expected to be partial) for later calling.
        """
        self.metadata = metadata
        self.metrics_cfg = kwargs  # All extra kwargs are metric definitions.

    def on_dr_end(self, dataset: any, embeddings: np.ndarray):
        logger.info("Computing additional metrics on DR result...")
        results = {}
        for metric_name, metric_fn in self.metrics_cfg.items():
            try:
                metric_value = metric_fn(dataset=dataset, embeddings=embeddings)
                logger.info(f"{metric_name}: {metric_value:.4f}")
                results[metric_name] = metric_value
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
        safe_wandb_log(results)
        return results
