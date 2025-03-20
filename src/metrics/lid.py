import numpy as np
from sklearn.neighbors import NearestNeighbors


def LocalIntrinsicDimensionality(dataset, embeddings: np.ndarray, k: int = 20) -> float:
    """
    Compute the mean Local Intrinsic Dimensionality (LID) for the embedding.

    Parameters:
      - dataset: Provided for protocol compliance.
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - k: The number of nearest neighbors to consider.

    Returns:
      - A float representing the mean LID.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = distances[:, 1:]  # Exclude the self-distance.
    r_k = distances[:, -1]
    lid_values = -k / np.sum(np.log(distances / r_k[:, None] + 1e-10), axis=1)
    return np.mean(lid_values)
