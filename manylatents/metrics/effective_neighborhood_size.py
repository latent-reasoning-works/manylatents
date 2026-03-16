import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["effective_neighborhood_size", "effective_k", "k_eff"],
    default_params={},
    description="Per-point effective neighborhood size from method's internal graph weights",
)
def EffectiveNeighborhoodSize(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    cache: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Per-point effective neighborhood size via participation ratio of edge weights.

    For each point, computes k_eff = (sum w)^2 / sum(w^2) from the row of the
    method's internal affinity/transition matrix. Equals k for uniform weights,
    approaches 1 when all weight concentrates on a single neighbor.

    Requires a fitted module that exposes affinity_matrix().

    Args:
        embeddings: (n_samples, n_features) array. Used only for shape.
        dataset: Unused, kept for protocol compatibility.
        module: Fitted LatentModule with affinity_matrix() method.
        cache: Optional shared cache dict.

    Returns:
        Dict with summary scalars and per-point k_eff array.
    """
    if module is None:
        raise ValueError(
            "EffectiveNeighborhoodSize requires a fitted module with "
            "affinity_matrix(). Pass module= to the metric."
        )

    try:
        W = module.affinity_matrix(ignore_diagonal=True, use_symmetric=False)
    except NotImplementedError:
        raise ValueError(
            f"{module.__class__.__name__} does not expose an affinity_matrix. "
            "EffectiveNeighborhoodSize requires access to the method's internal graph."
        )

    # Participation ratio per row: (sum w)^2 / sum(w^2)
    row_sum = np.array(W.sum(axis=1)).ravel()
    # Handle sparse matrices
    if hasattr(W, 'toarray'):
        W_dense = W.toarray()
    else:
        W_dense = np.asarray(W)

    row_sum_sq = np.array((W_dense ** 2).sum(axis=1)).ravel()

    # Guard against zero rows (isolated points)
    k_eff = np.where(
        row_sum_sq > 0,
        (row_sum ** 2) / row_sum_sq,
        0.0,
    )

    result = {
        "mean_k_eff": float(np.mean(k_eff)),
        "median_k_eff": float(np.median(k_eff)),
        "std_k_eff": float(np.std(k_eff)),
        "min_k_eff": float(np.min(k_eff)),
        "max_k_eff": float(np.max(k_eff)),
        "k_eff": k_eff,
    }

    logger.info(
        f"EffectiveNeighborhoodSize: mean={result['mean_k_eff']:.2f}, "
        f"median={result['median_k_eff']:.2f}, "
        f"std={result['std_k_eff']:.2f}"
    )
    return result
