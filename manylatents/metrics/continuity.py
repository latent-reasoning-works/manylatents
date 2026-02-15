from typing import Optional, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.utils.metrics import compute_knn


def Continuity(embeddings: np.ndarray,
               dataset: Optional[object] = None,
               module: Optional[LatentModule] = None,
               n_neighbors: int = 25,
               metric: str = 'euclidean',
               return_per_sample: bool = False,
               normalize: bool = True,
               adjust_for_random: bool = False,
               cache: Optional[dict] = None,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute (normalized and optionally adjusted) continuity between high-D and embedded space.

    Uses compute_knn (FAISS-accelerated) instead of pairwise_distances for O(n*k) vs O(n^2).

    Parameters:
    - embeddings: low-dimensional embedding (n_samples, d)
    - dataset: object with .data attribute (high-dimensional data, n_samples, D)
    - n_neighbors: size of the neighborhood (K')
    - metric: distance metric (unused — compute_knn uses L2)
    - return_per_sample: if True, also return pointwise overlap values per sample
    - normalize: if True, divide overlap by n_neighbors
    - adjust_for_random: if True, subtract expected random overlap
    - cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
    - continuity_score: mean continuity (scalar)
    - (optional) pointwise_continuity: array of shape (n_samples,)
    """
    X_high = dataset.data
    X_low = embeddings
    n = X_high.shape[0]

    _, knn_high = compute_knn(X_high, k=n_neighbors, include_self=False, cache=cache)
    _, knn_low = compute_knn(X_low, k=n_neighbors, include_self=False, cache=cache)

    # Compute pointwise neighborhood overlaps
    pointwise_overlap = np.array([
        len(np.intersect1d(knn_high[i], knn_low[i], assume_unique=True))
        for i in range(n)
    ])

    # Normalize to [0, 1]
    if normalize:
        pointwise_continuity = pointwise_overlap / n_neighbors
    else:
        pointwise_continuity = pointwise_overlap.astype(float)

    # Adjust for expected random overlap
    if adjust_for_random:
        pointwise_continuity -= n_neighbors / (n - 1)

    # Mean continuity score
    continuity_score = float(np.mean(pointwise_continuity))

    if return_per_sample:
        return continuity_score, pointwise_continuity
    else:
        return continuity_score
