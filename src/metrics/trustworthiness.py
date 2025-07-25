from typing import Optional

import numpy as np
import torch
from sklearn.manifold import trustworthiness as sk_trustworthiness

from src.algorithms.latent_module_base import LatentModule


def Trustworthiness(embeddings: np.ndarray, 
                    dataset: object, 
                    module: Optional[LatentModule] = None,
                    n_neighbors: int = 25, 
                    metric: str = 'euclidean') -> float:
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
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        
    return sk_trustworthiness(
        X=dataset.data, 
        X_embedded=embeddings, 
        n_neighbors=n_neighbors, 
        metric=metric
    )
