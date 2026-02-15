from typing import Optional

import numpy as np
from sklearn.manifold import trustworthiness as sk_trustworthiness

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


@register_metric(
    aliases=["trustworthiness", "trust"],
    default_params={"n_neighbors": 25},
    description="Trustworthiness of embedding (preservation of local structure)",
)
def Trustworthiness(embeddings: np.ndarray,
                    dataset: object,
                    module: Optional[LatentModule] = None,
                    n_neighbors: int = 25,
                    metric: str = 'euclidean',
                    cache: Optional[dict] = None) -> float:
    """
    Compute the trustworthiness of an embedding.

    Parameters:
      - dataset: An object with an attribute 'data' (the high-dimensional data).
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - n_neighbors: The number of neighbors to consider.
      - metric: The distance metric to use.
      - cache: Optional shared cache dict (unused — sklearn handles its own kNN).

    Returns:
      - A float representing the trustworthiness score.
    """
    return sk_trustworthiness(
        X=dataset.data,
        X_embedded=embeddings,
        n_neighbors=n_neighbors,
        metric=metric
    )
