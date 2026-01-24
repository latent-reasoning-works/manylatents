import logging
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric

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

    Returns:
        float or np.ndarray: average or per-sample PR.
    """
    if _knn_cache is not None:
        # Use precomputed kNN indices, slice to required k
        _, all_idx = _knn_cache
        # all_idx includes self at index 0, slice [0:n_neighbors+1] to get self + k neighbors
        all_idx = all_idx[:, :n_neighbors + 1]
    else:
        # build knn graph (include the point itself by doing k+1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
        _, all_idx = nbrs.kneighbors(embeddings)

    pr_list = []
    for idx in all_idx:
        neigh_pts = embeddings[idx[1:]]      # drop the query point itself
        centered = neigh_pts - neigh_pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)

        # get eigenvalues via SVD
        eigs = np.linalg.svd(cov, compute_uv=False)
        total = eigs.sum()
        if total <= 0:
            pr = 0.0
        else:
            pr = (total * total) / np.sum(eigs * eigs)

        pr_list.append(pr)

    pr_arr = np.array(pr_list, dtype=float)

    if return_per_sample:
        logger.info(f"ParticipationRatio: per-sample PR, mean={pr_arr.mean():.3f}")
        return pr_arr

    avg_pr = float(np.nanmean(pr_arr))
    logger.info(f"ParticipationRatio: average PR = {avg_pr:.3f}")
    return avg_pr
