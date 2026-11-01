"""Trustworthiness metric using shared kNN cache.

T(k) = 1 - (2 / (nk(2n-3k-1))) * sum_i sum_{j in U_k(i)} (r(i,j) - k)

U_k(i) = false neighbors: in embedding k-NN but not in original k-NN.
r(i,j) = rank of j w.r.t. i in original-space distances.
"""
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.metrics import compute_knn

#: Feature columns per Gram update, and rows per rank count; both bound peak memory only.
_FEATURE_BLOCK = 2048
_ROW_BLOCK = 256


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
    # For n, k > 0, n*k*(2*n - 3*k - 1) > 0 iff 3*k < 2*n - 1.
    # Thus integer k <= (2*n - 2)//3: floor((2*n - 1)/3), except
    # when that limit is integral, where equality gives zero and we subtract 1.
    if (not isinstance(k, (int, np.integer)) or isinstance(k, bool)
            or k <= 0 or 3 * k >= 2 * n - 1):
        raise MeasurementUnavailable(
            f"Trustworthiness requires integer 0 < n_neighbors < (2 * n_samples - 1) / 3; got k={k}, n={n}"
        )
    if X_low.shape[0] != n:
        raise MeasurementUnavailable("Trustworthiness requires matching sample counts")

    _, knn_high = compute_knn(X_high, k=k, include_self=False, cache=cache)
    _, knn_low = compute_knn(X_low, k=k, include_self=False, cache=cache)

    # Only the ranks of the false neighbours enter the sum, so the full rank matrix (cdist on
    # one thread, then two argsorts of n x n) is not needed: for each pair, r(i, j) is the
    # number of points nearer to i than j, self included. Squared distances come from the
    # Gram matrix, built in float64 over feature blocks so a float32 input is not copied
    # whole. On 5,000 x 20,255 this was 92 s; the result is the same up to rank ties.
    X = np.asarray(X_high)
    gram = np.zeros((n, n))
    for start in range(0, X.shape[1], _FEATURE_BLOCK):
        block = np.asarray(X[:, start:start + _FEATURE_BLOCK], dtype=np.float64)
        gram += block @ block.T
    sq = np.diag(gram).copy()
    dist2 = sq[:, None] + sq[None, :] - 2.0 * gram
    del gram
    np.fill_diagonal(dist2, -np.inf)

    knn_low = np.asarray(knn_low)
    false = ~(knn_low[:, :, None] == np.asarray(knn_high)[:, None, :]).any(axis=2)
    penalty = 0.0
    for start in range(0, n, _ROW_BLOCK):
        rows = slice(start, start + _ROW_BLOCK)
        d = dist2[rows]
        pair = np.take_along_axis(d, knn_low[rows], axis=1)
        rank = (d[:, None, :] < pair[:, :, None]).sum(axis=2)
        penalty += float(((rank - k) * false[rows]).sum())

    normalizer = n * k * (2 * n - 3 * k - 1)
    return float(1.0 - (2.0 / normalizer) * penalty)
