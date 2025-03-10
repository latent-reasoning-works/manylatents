import logging

import numpy as np
from sklearn.manifold import trustworthiness

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback

logger = logging.getLogger(__name__)

class AdditionalMetrics(DimensionalityReductionCallback):
    def __init__(self, **kwargs):
        """
        Hydra will pass in metric definitions as keyword arguments.
        The 'metadata' key is popped separately.
        All remaining keys are assumed to be metric configurations.
        """
        self.metadata = kwargs.pop("metadata", None)
        self.metrics_cfg = kwargs  # remaining keys are metric configs

    def on_dr_end(self, dataset: any, embeddings: np.ndarray):
        original = dataset.full_data  # retrieve the original data from the dataset
        results = {}
        for metric_name, cfg in self.metrics_cfg.items():
            try:
                if metric_name == "trustworthiness":
                    score = trustworthiness(original, embeddings, **cfg)
                else:
                    logger.warning(f"Metric '{metric_name}' is not implemented. Skipping.")
                    continue

                logger.info(f"{metric_name}: {score:.4f}")
                results[metric_name] = score
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
        return results
