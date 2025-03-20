import numpy as np
from ripser import ripser


def PersistentHomology(dataset, embeddings: np.ndarray, homology_dim: int = 1, persistence_threshold: float = 0.1) -> float:
    """
    Compute a persistent homology metric for the embedding.

    This function uses ripser to compute the persistent diagram and then counts
    the number of features in the specified homology dimension whose persistence
    (death - birth) exceeds a given threshold.

    Parameters:
      - dataset: Provided for protocol compliance.
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - homology_dim: Homology dimension to analyze (e.g., 0 for connected components, 1 for loops).
      - persistence_threshold: Minimum persistence for a feature to be counted.

    Returns:
      - Number of persistent features (float).

    Note:
      - Requires ripser, which is included in the provided environment.
    """

    diagrams = ripser(embeddings)['dgms']
    features = diagrams[homology_dim]
    persistence = features[:, 1] - features[:, 0]
    count = np.sum(persistence > persistence_threshold)
    return float(count)
