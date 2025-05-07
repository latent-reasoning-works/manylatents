from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.algorithms.dimensionality_reduction import DimensionalityReductionModule

def KNNPreservation(
    dataset,
    embeddings: np.ndarray,
    n_neighbors: int = 10,
    metric: str = 'euclidean',
    module: Optional[DimensionalityReductionModule] = None
) -> float:
    """
    Compute the average k-NN preservation between high-dimensional data and its low-dimensional embedding.

    This metric reflects the proportion of shared neighbors for each point between the
    high-dimensional space and the low-dimensional embedding.

    Parameters:
        - dataset: An object with an attribute 'original_data' (the high-dimensional data).
        - embeddings (np.ndarray): Low-dimensional embeddings of shape (n_samples, n_components).
        - n_neighbors (int): Number of neighbors to consider for kNN graph.
        - metric (str): Distance metric to use for NearestNeighbors.

    Returns:
        - float: Mean k-NN overlap score (between 0 and 1).
    """
    # Fit k-NN models
    knn_high = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(dataset.data)
    knn_low = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(embeddings)

    # Get neighbor indices
    neighbors_high = knn_high.kneighbors(return_distance=False)
    neighbors_low = knn_low.kneighbors(return_distance=False)

    # Compute average neighbor overlap
    total_overlap = 0
    n_samples = dataset.data.shape[0]

    for i in range(n_samples):
        overlap = len(set(neighbors_high[i]) & set(neighbors_low[i]))
        total_overlap += overlap

    knn_preservation_score = total_overlap / (n_samples * n_neighbors)
    return knn_preservation_score