"""Hydra-free metric evaluation API.

Resolves metrics by name from the registry, manages cache sharing,
and pre-warms kNN/eigenvalues. This is the direct Python interface
for agents and analysis scripts.

The Hydra experiment engine (evaluate_outputs in experiment.py) keeps
its own Hydra-based metric loop for now. Future refactoring may converge
both paths, but evaluate_outputs currently does NOT delegate to
evaluate_metrics — the ColormapInfo unpacking and DictConfig partial
instantiation are Hydra-specific.
"""
import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import get_metric

logger = logging.getLogger(__name__)


def _flatten_metric_result(metric_name: str, raw_result: Any) -> dict[str, Any]:
    """Flatten a metric result into a dict of name->value pairs."""
    if isinstance(raw_result, dict):
        return {f"{metric_name}.{k}": v for k, v in raw_result.items()}
    return {metric_name: raw_result}


def evaluate_metrics(
    embeddings: np.ndarray,
    *,
    metrics: list[str],
    module: Optional[object] = None,
    dataset: Optional[object] = None,
    cache: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate named metrics on embeddings without Hydra.

    Args:
        embeddings: (n_samples, n_features) array.
        metrics: List of metric names or aliases from the registry.
        module: Optional fitted LatentModule (for module-context metrics).
        dataset: Optional dataset object (for dataset-context metrics).
        cache: Optional shared cache dict. Created internally if None.
            Pass a dict to share kNN/eigenvalue caches across calls.
        **kwargs: Additional params forwarded to all metrics.

    Returns:
        Flat dict of metric results. Dict-returning metrics are flattened
        with dotted keys (e.g., "k_eff.mean_k_eff").
    """
    if cache is None:
        cache = {}

    # Pre-warm cache based on metric requirements.
    # Import is lazy (inside function body) to avoid circular dependency:
    # evaluate.py -> experiment.py -> evaluate.py (_flatten_metric_result).
    from manylatents.experiment import prewarm_cache
    prewarm_cache(metrics, embeddings, dataset, module, cache=cache)

    results: dict[str, Any] = {}
    for name in metrics:
        spec = get_metric(name)
        raw_result = spec(
            embeddings=embeddings,
            dataset=dataset,
            module=module,
            cache=cache,
            **kwargs,
        )
        results.update(_flatten_metric_result(name, raw_result))

    return results
