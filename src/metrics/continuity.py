from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule


def Continuity(embeddings: np.ndarray, 
               dataset: Optional[object] = None, 
               module: Optional[DimensionalityReductionModule] = None,
               n_neighbors: int = 25,
               metric: str = 'euclidean',
) -> float:
    """
    Compute the continuity metric of embeddings, comparing neighborhood preservation 
    from embedding space to the original space.
    """
    X_high = dataset.data
    X_low = embeddings

    # Compute pairwise distances
    dist_high = pairwise_distances(X_high, metric=metric)
    dist_low = pairwise_distances(X_low, metric=metric)

    # Rank matrices
    rank_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)
    rank_low = np.argsort(np.argsort(dist_low, axis=1), axis=1)

    N = X_high.shape[0]
    continuity_sum = 0.0

    for i in range(N):
        low_neighbors = np.where(rank_low[i] <= n_neighbors)[0]
        high_ranks_of_low_neighbors = rank_high[i, low_neighbors]
        continuity_sum += np.sum(high_ranks_of_low_neighbors - n_neighbors)

    continuity_score = 1 - (2 / (N * n_neighbors * (2 * N - 3 * n_neighbors - 1))) * continuity_sum
    return float(continuity_score)
