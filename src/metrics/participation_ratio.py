import logging
from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def ParticipationRatio(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    return_per_sample: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute the local Participation Ratio (PR) for each sample’s neighborhood.

    PR = (sum_i λ_i)^2 / sum_i (λ_i^2)
    where λ_i are the eigenvalues of the local covariance.

    Args:
        dataset:      (unused) kept for Protocol compatibility.
        embeddings:   (n_samples, n_features) array (always numpy).
        n_neighbors:  how many neighbors (k) to use.
        return_per_sample:
                      if True, returns array of shape (n_samples,) with each sample’s PR;
                      else returns the average PR (float).

    Returns:
        float or np.ndarray: average or per-sample PR.
    """
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
