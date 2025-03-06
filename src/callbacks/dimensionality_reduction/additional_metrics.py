import logging

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
        # All remaining keys are assumed to be metric configs.
        self.metrics_cfg = kwargs

    def on_dr_end(self, original, embedded):
        """
        Compute additional metrics. Because different metrics expect different inputs,
        we dispatch the call for each metric based on its name.
        
        For example:
          - For 'trustworthiness', call with X and X_embedded.
          - For 'silhouette', perhaps you need X and labels (which might be stored in metadata).
          - For unknown metrics, log a warning.
        """
        results = {}
        for metric_name, cfg in self.metrics_cfg.items():
            try:
                if metric_name == "trustworthiness":
                    score = trustworthiness(original, embedded, **cfg)
                else:
                    # For additional metrics, add their dispatch logic here.
                    logger.warning(f"Metric '{metric_name}' is not implemented. Skipping.")
                    continue

                logger.info(f"{metric_name}: {score:.4f}")
                results[metric_name] = score
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
        return results


