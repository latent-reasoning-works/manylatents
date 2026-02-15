import logging
from typing import Optional, Union

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["local_intrinsic_dim", "lid", "intrinsic_dim"],
    default_params={"return_per_sample": False},
    description="Mean local intrinsic dimensionality of the embedding",
)
def LocalIntrinsicDimensionality(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: int = 20,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the Local Intrinsic Dimensionality (LID) for the embedding.

    Uses maximum likelihood estimation based on k-NN distances.
    kNN is computed via FAISS when available (~10-50x faster), sklearn otherwise.

    Parameters:
      - embeddings: A numpy array representing the embeddings.
      - dataset: Provided for protocol compliance (unused).
      - module: Provided for protocol compliance (unused).
      - k: The number of nearest neighbors to consider.
      - return_per_sample: If True, return per-sample LID values; else return mean.
      - cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
      - float: Mean LID (if return_per_sample=False)
      - np.ndarray: Per-sample LID values (if return_per_sample=True)
    """
    distances, _ = compute_knn(embeddings, k=k, include_self=False, cache=cache)

    # LID computation: MLE estimator
    r_k = distances[:, -1]
    r_k = np.maximum(r_k, 1e-10)  # prevent division by zero (duplicate embeddings)
    lid_values = -k / np.sum(np.log(distances / r_k[:, None] + 1e-10), axis=1)

    if return_per_sample:
        logger.info(f"LocalIntrinsicDimensionality: per-sample LID, mean={np.mean(lid_values):.3f}")
        return lid_values

    mean_lid = float(np.mean(lid_values))
    logger.info(f"LocalIntrinsicDimensionality: Computed mean LID = {mean_lid:.3f}")
    return mean_lid
