import logging

import numpy as np

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback

logger = logging.getLogger(__name__)

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
        return results
