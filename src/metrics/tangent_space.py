import numpy as np
from sklearn.neighbors import NearestNeighbors


def TangentSpaceApproximation(dataset, embeddings: np.ndarray, n_neighbors: int = 10, variance_threshold: float = 0.95) -> float:
    """
    Estimate the local manifold dimension by approximating the tangent space via PCA.

    For each point, determine the number of principal components required to explain
    a given fraction of the variance and return the average estimated dimension.

    Parameters:
      - dataset: Provided for protocol compliance.
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - n_neighbors: Number of neighbors to consider.
      - variance_threshold: Cumulative explained variance threshold for determining the dimension.

    Returns:
      - Average local dimension (float).
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    dims = []
    for idx in indices:
        local_points = embeddings[idx[1:]]  # Exclude the point itself.
        local_centered = local_points - np.mean(local_points, axis=0)
        # Perform SVD on the local neighborhood.
        _, s, _ = np.linalg.svd(local_centered, full_matrices=False)
        variance_explained = (s ** 2) / np.sum(s ** 2)
        cum_variance = np.cumsum(variance_explained)
        local_dim = np.searchsorted(cum_variance, variance_threshold) + 1
        dims.append(local_dim)
    return float(np.mean(dims))
