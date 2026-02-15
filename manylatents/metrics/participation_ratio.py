import logging
from typing import Optional, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["participation_ratio", "pr"],
    default_params={"return_per_sample": False},
    description="Local participation ratio measuring effective dimensionality",
)
def ParticipationRatio(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the local Participation Ratio (PR) for each sample's neighborhood.

    PR = (sum_i lambda_i)^2 / sum_i (lambda_i^2)
    where lambda_i are the eigenvalues of the local covariance.

    Args:
        embeddings:   (n_samples, n_features) array (always numpy).
        dataset:      (unused) kept for Protocol compatibility.
        module:       (unused) kept for Protocol compatibility.
        n_neighbors:  how many neighbors (k) to use.
        return_per_sample:
                      if True, returns array of shape (n_samples,) with each sample's PR;
                      else returns the average PR (float).
        cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
        float or np.ndarray: average or per-sample PR.
    """
    n_samples = embeddings.shape[0]

    _, all_idx = compute_knn(embeddings, k=n_neighbors, include_self=True, cache=cache)

    k = all_idx.shape[1] - 1
    chunk_size = max(1, min(10_000, int(2e9 / (k * embeddings.shape[1] * 4))))

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[all_idx[start:end, 1:]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

    s = np.concatenate(sv_chunks, axis=0)

    # Eigenvalues of covariance are proportional to s^2
    s2 = s * s
    total = s2.sum(axis=1)
    sum_sq = (s2 * s2).sum(axis=1)
    pr_arr = np.where(total > 0, total * total / sum_sq, 0.0)

    if return_per_sample:
        logger.info(f"ParticipationRatio: per-sample PR, mean={pr_arr.mean():.3f}")
        return pr_arr

    avg_pr = float(np.nanmean(pr_arr))
    logger.info(f"ParticipationRatio: average PR = {avg_pr:.3f}")
    return avg_pr
