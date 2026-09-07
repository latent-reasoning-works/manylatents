"""Trustworthiness metric using shared kNN cache.

T(k) = 1 - (2 / (nk(2n-3k-1))) * sum_i sum_{j in U_k(i)} (r(i,j) - k)

U_k(i) = false neighbors: in embedding k-NN but not in original k-NN.
r(i,j) = rank of j w.r.t. i in original-space distances.
"""
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.metrics import compute_knn


@register_metric(
    aliases=["trustworthiness", "trust"],
    default_params={"n_neighbors": 25},
    description="Trustworthiness of embedding (preservation of local structure)",
)
def Trustworthiness(embeddings: np.ndarray,
                    dataset=None,
                    module: Optional[LatentModule] = None,
                    n_neighbors: int = 25,
                    metric: str = 'euclidean',
                    cache: Optional[dict] = None) -> float:
    """
    Compute the trustworthiness of an embedding.

    Parameters:
      - embeddings: Low-dimensional embeddings (n_samples, d).
      - dataset: Object with .data attribute (high-dimensional data).
      - module: LatentModule instance (unused).
      - n_neighbors: Number of neighbors to consider.
      - metric: Distance metric (unused — compute_knn uses L2).
      - cache: Shared cache dict passed through to compute_knn().

    Returns:
      - A float representing the trustworthiness score.
    """
    if dataset is None or getattr(dataset, "data", None) is None:
        raise MeasurementUnavailable("Trustworthiness requires dataset.data")
    X_high = dataset.data
    X_low = embeddings
    n = X_high.shape[0]
    k = n_neighbors
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 0 < k < n / 2:
        raise MeasurementUnavailable(
            f"Trustworthiness requires 0 < n_neighbors < n_samples / 2; got k={k}, n={n}"
        )
    if X_low.shape[0] != n:
        raise MeasurementUnavailable("Trustworthiness requires matching sample counts")

    _, knn_high = compute_knn(X_high, k=k, include_self=False, cache=cache)
    _, knn_low = compute_knn(X_low, k=k, include_self=False, cache=cache)

    knn_high_sets = [set(knn_high[i]) for i in range(n)]

    # Rank matrix from original-space distances
    dist_high = cdist(X_high, X_high, metric='euclidean')
    np.fill_diagonal(dist_high, -np.inf)
    rank_matrix = np.argsort(np.argsort(dist_high, axis=1), axis=1)

    penalty = 0.0
    for i in range(n):
        for j in knn_low[i]:
            if j not in knn_high_sets[i]:
                penalty += rank_matrix[i, j] - k

    normalizer = n * k * (2 * n - 3 * k - 1)
    return float(1.0 - (2.0 / normalizer) * penalty)
