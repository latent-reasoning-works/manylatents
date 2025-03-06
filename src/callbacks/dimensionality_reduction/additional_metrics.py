import logging

import hydra

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback

logger = logging.getLogger(__name__)


class AdditionalMetrics(DimensionalityReductionCallback):
    def __init__(self, **kwargs):
        """
        - `metrics_cfg` is a DictConfig that contains all your metric definitions.
        - Because `_recursive_: false` is set at `metrics_cfg`, Hydra won't
        instantiate `trustworthiness` prematurely.
        """
        self.metadata = kwargs.pop("metadata", None)
        # All remaining keys are assumed to be metric configs.
        self.metrics_cfg = kwargs

    def on_dr_end(self, original, embedded):
        # Build a context dictionary with multiple common names.
        # This allows metrics that expect, e.g., 'X' and 'X_embedded'
        # to work alongside those expecting 'original' and 'embedded'.
        context = {
            "X": original,
            "X_embedded": embedded,
            "original": original,
            "embedded": embedded,
        }
        results = {}
        for metric_name, cfg in self.metrics_cfg.items():
            # Use keyword arguments so that the target function receives its expected names.
            score = hydra.utils.call(cfg, **context)
            logger.info(f"{metric_name}: {score:.4f}")
            results[metric_name] = score
        return results

