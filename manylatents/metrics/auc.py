"""Area Under ROC Curve metric.

Computes AUC from predictions (embeddings) and labels from dataset.
Used with ClassifierModule which outputs P(y=1|x) as its "embedding".

Example:
    >>> clf = ClassifierModule()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.transform(X_test)
    >>> auc_metric = AUC()
    >>> result = auc_metric(predictions, test_dataset)
    >>> print(result)  # {"auc": 0.85}
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["auc", "roc_auc"],
    default_params={},
    description="Area Under ROC Curve for binary classification",
)
def AUC(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    cache: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Compute Area Under ROC Curve from predictions and labels.

    Parameters:
        embeddings: Predictions of shape (N,) or (N, 1). Should be P(y=1|x).
        dataset: Dataset with get_labels() method returning binary labels.
        module: Provided for protocol compliance (unused).
        cache: Optional shared cache dict (unused).

    Returns:
        Dict with "auc" key containing the AUC score.

    Raises:
        ValueError: If dataset has no get_labels() method or labels are invalid.
    """
    # Flatten predictions if needed
    predictions = embeddings.flatten()

    # Get labels from dataset
    if dataset is None or not hasattr(dataset, "get_labels"):
        logger.warning("AUC metric requires a dataset with get_labels(). Skipping.")
        return {"auc": float("nan")}

    labels = dataset.get_labels()
    if isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    # Validate
    if len(predictions) != len(labels):
        logger.warning(
            "AUC: predictions (%d) and labels (%d) length mismatch. Skipping.",
            len(predictions), len(labels),
        )
        return {"auc": float("nan")}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in labels: {unique_labels}. AUC undefined.")
        return {"auc": float("nan")}

    # Compute AUC
    auc = roc_auc_score(labels, predictions)
    logger.info(f"AUC: Computed AUC = {auc:.4f}")

    return {"auc": float(auc)}
