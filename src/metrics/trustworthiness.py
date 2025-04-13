import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness


def Trustworthiness(dataset, embeddings: np.ndarray, n_neighbors: int = 5, metric: str = 'euclidean') -> float:
    """
    Compute the trustworthiness of an embedding.

    Parameters:
      - dataset: An object with an attribute 'original_data' (the high-dimensional data).
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - n_neighbors: The number of neighbors to consider.
      - metric: The distance metric to use.

    Returns:
      - A float representing the trustworthiness score.
    """
    return sk_trustworthiness(
        X=dataset.data, 
        X_embedded=embeddings, 
        n_neighbors=n_neighbors, 
        metric=metric
    )
