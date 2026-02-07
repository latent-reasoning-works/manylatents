from typing import Optional, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.utils.metrics import compute_knn


def KNNPreservation(
    embeddings: np.ndarray,
    dataset,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 10,
    metric: str = 'euclidean',
    return_per_sample: bool = False,
    _knn_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the average k-NN preservation between high-dimensional data and its low-dimensional embedding.

    This metric reflects the proportion of shared neighbors for each point between the
    high-dimensional space and the low-dimensional embedding.
    kNN is computed via FAISS when available (~10-50x faster), sklearn otherwise.

    Parameters:
        embeddings: Low-dimensional embeddings of shape (n_samples, n_components).
        dataset: An object with an attribute 'data' (the high-dimensional data).
        module: (unused) kept for Protocol compatibility.
        n_neighbors: Number of neighbors to consider for kNN graph.
        metric: Distance metric to use for NearestNeighbors.
        return_per_sample: If True, return per-sample overlap scores.
        _knn_cache: Optional precomputed (distances, indices) for embeddings.
            Note: Only used for embeddings; high-dim kNN is always computed.

    Returns:
        float: Mean k-NN overlap score (between 0 and 1).
        np.ndarray: Per-sample scores if return_per_sample=True.
    """
    n_samples = dataset.data.shape[0]

    # High-dimensional kNN (always computed - no cache for high-dim)
    _, neighbors_high = compute_knn(dataset.data, k=n_neighbors, include_self=False)

    # Low-dimensional (embedding) kNN - use cache if available
    if _knn_cache is not None:
        _, indices = _knn_cache
        # Slice to required k (indices includes self at 0)
        neighbors_low = indices[:, 1:n_neighbors + 1]
    else:
        _, neighbors_low = compute_knn(embeddings, k=n_neighbors, include_self=False)

    # Compute per-sample neighbor overlap
    overlap_scores = np.array([
        len(set(neighbors_high[i]) & set(neighbors_low[i])) / n_neighbors
        for i in range(n_samples)
    ])

    if return_per_sample:
        return overlap_scores

    return float(np.mean(overlap_scores))