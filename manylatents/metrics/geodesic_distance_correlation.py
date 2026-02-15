"""Geodesic Distance Correlation metric.

Correlation between ground truth geodesic distances and embedding pairwise distances.
Supports Spearman and Kendall tau correlation types.
"""
import logging
import warnings
from typing import Optional

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import pairwise_distances

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["geodesic_correlation", "geodesic_distance_correlation"],
    default_params={"correlation_type": "spearman"},
    description="Correlation between geodesic and embedded distances",
)
def GeodesicDistanceCorrelation(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    correlation_type: str = "spearman",
    cache: Optional[dict] = None,
) -> float:
    """Compute correlation between geodesic and embedding pairwise distances.

    Args:
        embeddings: (n_samples, n_features) embedding.
        dataset: Dataset with get_gt_dists() method.
        module: LatentModule (unused).
        correlation_type: "spearman" or "kendall".

    Returns:
        float: Correlation coefficient, or nan if ground truth unavailable.
    """
    if dataset is None or not hasattr(dataset, "get_gt_dists") or not callable(dataset.get_gt_dists):
        warnings.warn("GeodesicDistanceCorrelation: no get_gt_dists() available.", RuntimeWarning)
        return float("nan")

    try:
        gt_dists = dataset.get_gt_dists()
    except Exception:
        return float("nan")

    if gt_dists is None:
        return float("nan")

    emb_dists = pairwise_distances(embeddings, metric="euclidean")

    # Extract upper triangle (avoid diagonal and duplicates)
    triu_idx = np.triu_indices_from(gt_dists, k=1)
    gt_flat = gt_dists[triu_idx]
    emb_flat = emb_dists[triu_idx]

    if correlation_type == "spearman":
        corr, _ = spearmanr(gt_flat, emb_flat)
    elif correlation_type == "kendall":
        corr, _ = kendalltau(gt_flat, emb_flat)
    else:
        raise ValueError(f"Unknown correlation_type: {correlation_type}")

    result = float(corr)
    logger.info(f"GeodesicDistanceCorrelation ({correlation_type}): {result:.4f}")
    return result
