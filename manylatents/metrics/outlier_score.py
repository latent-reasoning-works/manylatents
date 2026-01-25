"""Outlier Score metric using Local Outlier Factor.

Computes outlier scores for embeddings. Hypothesis: variants that deviate
from a shared cross-modal subspace (high outlier score) are enriched for
pathogenicity.

Example:
    >>> merger = MergingModule(strategy="concat_pca", target_dim=128)
    >>> embeddings = merger.fit_transform(dummy)
    >>> outlier_metric = OutlierScore(k=20)
    >>> result = outlier_metric(embeddings, dataset)
    >>> print(result)  # {"mean": 1.2, "std": 0.3, "auc": 0.72}
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["outlier", "lof", "outlier_score"],
    default_params={"k": 20, "return_scores": False},
    description="Outlier scores using Local Outlier Factor",
)
def OutlierScore(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: int = 20,
    return_scores: bool = False,
    _knn_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute outlier scores using Local Outlier Factor.

    Parameters:
        embeddings: Embedding array of shape (N, D).
        dataset: Optional dataset with get_labels() for AUC computation.
        module: Provided for protocol compliance (unused).
        k: Number of neighbors for LOF. Default 20.
        return_scores: If True, include per-sample scores in result.
        _knn_cache: Provided for protocol compliance (unused).

    Returns:
        Dict containing:
        - "mean": Mean outlier score
        - "std": Standard deviation of outlier scores
        - "auc": AUC of outlier scores vs labels (if dataset has labels)
        - "scores": Per-sample outlier scores (if return_scores=True)
    """
    # Ensure embeddings is 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    # Fit LOF
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit(embeddings)

    # Get outlier scores (negative_outlier_factor_ is negative, so negate)
    scores = -lof.negative_outlier_factor_

    result: Dict[str, Any] = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

    logger.info(f"OutlierScore: mean={result['mean']:.4f}, std={result['std']:.4f}")

    # Compute AUC if labels available
    if dataset is not None and hasattr(dataset, "get_labels"):
        try:
            labels = dataset.get_labels()
            if isinstance(labels, (list, tuple)):
                labels = np.array(labels)

            unique_labels = np.unique(labels)
            if len(unique_labels) >= 2:
                auc = roc_auc_score(labels, scores)
                result["auc"] = float(auc)
                logger.info(f"OutlierScore: AUC (outlier vs pathogenic) = {auc:.4f}")
            else:
                logger.warning(f"Only one class present: {unique_labels}. Skipping AUC.")
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")

    if return_scores:
        result["scores"] = scores

    return result
