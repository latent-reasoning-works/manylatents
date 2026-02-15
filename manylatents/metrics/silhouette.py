"""Silhouette Score metric.

Uses torchdr.silhouette_score when available for GPU acceleration,
falls back to sklearn.metrics.silhouette_score.
"""
import logging
import warnings
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["silhouette", "silhouette_score"],
    default_params={"metric": "euclidean"},
    description="Silhouette score for cluster separation in embedding",
)
def SilhouetteScore(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    metric: str = "euclidean",
    cache: Optional[dict] = None,
) -> float:
    """Compute silhouette coefficient of embedding w.r.t. cluster labels.

    Args:
        embeddings: (n_samples, n_features) embedding array.
        dataset: Dataset with .metadata containing cluster labels.
        module: LatentModule (unused).
        metric: Distance metric for silhouette computation.

    Returns:
        float: Silhouette score in [-1, 1], or nan if labels unavailable.
    """
    labels = _extract_labels(dataset)
    if labels is None:
        warnings.warn("SilhouetteScore: no labels available, returning nan.", RuntimeWarning)
        return float("nan")

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        warnings.warn("SilhouetteScore: fewer than 2 clusters, returning nan.", RuntimeWarning)
        return float("nan")

    try:
        from manylatents.utils.backend import check_torchdr_available

        if check_torchdr_available():
            import torch
            from torchdr import silhouette_score

            X_t = torch.from_numpy(embeddings).float()
            labels_t = torch.from_numpy(labels.astype(np.int64))
            score = silhouette_score(X_t, labels_t, metric=metric)
            result = float(score)
            logger.info(f"SilhouetteScore (torchdr): {result:.4f}")
            return result
    except Exception:
        pass

    from sklearn.metrics import silhouette_score as sk_silhouette

    result = float(sk_silhouette(embeddings, labels, metric=metric))
    logger.info(f"SilhouetteScore (sklearn): {result:.4f}")
    return result


def _extract_labels(dataset: Optional[object]) -> Optional[np.ndarray]:
    """Extract labels from dataset."""
    if dataset is None:
        return None

    labels = getattr(dataset, "metadata", None)
    if labels is None and hasattr(dataset, "get_labels"):
        labels = dataset.get_labels()

    if labels is None:
        return None

    return np.asarray(labels)
