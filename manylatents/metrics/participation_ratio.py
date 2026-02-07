import logging
from typing import Dict, Optional, Tuple, Union

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
    _knn_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    _svd_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the local Participation Ratio (PR) for each sample's neighborhood.

    PR = (sum_i λ_i)^2 / sum_i (λ_i^2)
    where λ_i are the eigenvalues of the local covariance.

    Args:
        embeddings:   (n_samples, n_features) array (always numpy).
        dataset:      (unused) kept for Protocol compatibility.
        module:       (unused) kept for Protocol compatibility.
        n_neighbors:  how many neighbors (k) to use.
        return_per_sample:
                      if True, returns array of shape (n_samples,) with each sample's PR;
                      else returns the average PR (float).
        _knn_cache:   Optional (distances, indices) tuple from precomputed kNN.
                      Indices should be shape (n_samples, max_k+1) including self.
        _svd_cache:   Optional {k: singular_values} dict from precomputed SVD.
                      If provided and contains n_neighbors, skips SVD computation.

    Returns:
        float or np.ndarray: average or per-sample PR.
    """
    n_samples = embeddings.shape[0]

    # Try to use cached singular values
    if _svd_cache is not None and n_neighbors in _svd_cache:
        s = _svd_cache[n_neighbors]
        logger.debug(f"ParticipationRatio: using cached SVD for k={n_neighbors}")
    else:
        # Fall back to computing SVD inline
        if _knn_cache is not None:
            _, all_idx = _knn_cache
            all_idx = all_idx[:, :n_neighbors + 1]
        else:
            _, all_idx = compute_knn(embeddings, k=n_neighbors, include_self=True)

        k = all_idx.shape[1] - 1
        chunk_size = max(1, min(10_000, int(2e9 / (k * embeddings.shape[1] * 4))))

        sv_chunks = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            neigh = embeddings[all_idx[start:end, 1:]]
            centered = neigh - neigh.mean(axis=1, keepdims=True)
            sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

        s = np.concatenate(sv_chunks, axis=0)

    # Eigenvalues of covariance ∝ s²
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
