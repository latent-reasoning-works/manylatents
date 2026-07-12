import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


def _k_eff_common_kernel(
    embeddings: np.ndarray,
    k: int,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """Per-row participation ratio of a shared Gaussian kNN transition kernel.

    Builds a DR-method-agnostic kNN graph on embedding-space distances with
    adaptive (self-tuning) bandwidth ``sigma_i = d_i_k``, symmetrizes by union,
    row-normalizes to a stochastic matrix, and returns
    ``k_eff_i = 1 / sum_j P_ij^2``.

    Args:
        embeddings: (n_samples, n_features) float32 array (low-dim embedding).
        k: Neighborhood size (excluding self).
        cache: Optional shared cache dict (forwarded to compute_knn).

    Returns:
        (n_samples,) array of per-row k_eff values.
    """
    from manylatents.utils.knn import compute_knn

    emb = np.ascontiguousarray(np.asarray(embeddings), dtype=np.float32)
    n = emb.shape[0]

    # kNN, self included at index 0 (so neighbors columns are 1..k)
    distances, indices = compute_knn(emb, k=k, include_self=True, cache=cache)
    # neighbor distances (n, k), excluding self
    d = distances[:, 1:]
    idx = indices[:, 1:]

    # Adaptive bandwidth: distance to the k-th neighbor (last column).
    # Floor at float32 eps to avoid divide-by-zero on duplicate points.
    eps = float(np.finfo(np.float32).eps)
    sigma = np.maximum(d[:, -1], eps).astype(np.float64)  # (n,)

    # Gaussian weights on kNN edges: W_ij = exp(-d_ij^2 / (sigma_i * sigma_j))
    sigma_j = sigma[idx]  # (n, k)
    denom = sigma[:, None] * sigma_j  # (n, k)
    w = np.exp(-(d.astype(np.float64) ** 2) / denom)  # (n, k)

    # Scatter into dense (n, n). n ~ up to a few thousand in this pipeline; if
    # memory matters we can replace with scipy.sparse.coo_matrix, but dense is
    # simplest, matches the native path, and is O(n*k) to fill.
    W = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), d.shape[1])
    cols = idx.reshape(-1)
    W[rows, cols] = w.reshape(-1)

    # Union symmetrize: W = max(W, W^T)
    W = np.maximum(W, W.T)

    # Row-normalize to row-stochastic P
    row_sum = W.sum(axis=1)
    row_sum_safe = np.where(row_sum > 0, row_sum, 1.0)
    P = W / row_sum_safe[:, None]

    # Per-row participation ratio: 1 / sum_j P_ij^2
    sum_sq = (P ** 2).sum(axis=1)
    k_eff = np.where(sum_sq > 0, 1.0 / sum_sq, 0.0)
    return k_eff


@register_metric(
    aliases=["effective_neighborhood_size", "effective_k", "k_eff"],
    default_params={},
    description="Per-point effective neighborhood size from method's internal graph weights",
)
def EffectiveNeighborhoodSize(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    cache: Optional[dict] = None,
    mode: str = "native",
    k: int = 15,
) -> dict[str, Any]:
    """
    Per-point effective neighborhood size via participation ratio of edge weights.

    Two modes:

    - ``mode="native"`` (default, original behavior): reads the fitted module's
      internal affinity matrix via ``module.affinity(ignore_diagonal=True,
      use_symmetric=False)`` and computes ``k_eff_i = (sum w)^2 / sum(w^2)``
      on each row. Requires a module that exposes ``affinity()``. For methods
      whose affinity is a signed Gram matrix (PCA, MDS, Sammon) this is
      degenerate — use ``mode="common_kernel"`` for cross-family comparison.

    - ``mode="common_kernel"``: ignores ``module.affinity()`` entirely. Builds a
      kNN Gaussian transition kernel on the embedding (adaptive bandwidth
      ``sigma_i = d_i_k``, symmetrized by union, row-normalized), and returns
      the participation ratio ``1 / sum_j P_ij^2`` of each row. Works
      uniformly across DR families.

    Args:
        embeddings: (n_samples, n_features) array. Used in common_kernel mode.
            In native mode used only for shape.
        dataset: Unused, kept for protocol compatibility.
        module: Fitted LatentModule. Required in native mode; ignored in
            common_kernel mode.
        cache: Optional shared cache dict (forwarded to compute_knn).
        mode: ``"native"`` or ``"common_kernel"``.
        k: Neighborhood size for common_kernel mode (default 15). Ignored in
            native mode.

    Returns:
        Dict with summary scalars (``mean_k_eff``, ``median_k_eff``,
        ``std_k_eff``, ``min_k_eff``, ``max_k_eff``) and the per-point
        ``k_eff`` array.
    """
    if mode == "common_kernel":
        if embeddings is None:
            raise ValueError(
                "EffectiveNeighborhoodSize(mode='common_kernel') requires "
                "embeddings to build the shared kNN kernel."
            )
        k_eff = _k_eff_common_kernel(embeddings, k=int(k), cache=cache)

    elif mode == "native":
        if module is None:
            raise ValueError(
                "EffectiveNeighborhoodSize requires a fitted module with "
                "affinity_matrix(). Pass module= to the metric."
            )

        try:
            W = module.affinity(ignore_diagonal=True, use_symmetric=False)
        except NotImplementedError:
            raise ValueError(
                f"{module.__class__.__name__} does not expose an affinity_matrix. "
                "EffectiveNeighborhoodSize requires access to the method's internal graph."
            )

        # Participation ratio per row: (sum w)^2 / sum(w^2)
        row_sum = np.array(W.sum(axis=1)).ravel()
        # Handle sparse matrices
        if hasattr(W, 'toarray'):
            W_dense = W.toarray()
        else:
            W_dense = np.asarray(W)

        row_sum_sq = np.array((W_dense ** 2).sum(axis=1)).ravel()

        # Guard against zero rows (isolated points)
        k_eff = np.where(
            row_sum_sq > 0,
            (row_sum ** 2) / row_sum_sq,
            0.0,
        )
    else:
        raise ValueError(
            f"EffectiveNeighborhoodSize: unknown mode={mode!r}. "
            "Expected 'native' or 'common_kernel'."
        )

    result = {
        "mean_k_eff": float(np.mean(k_eff)),
        "median_k_eff": float(np.median(k_eff)),
        "std_k_eff": float(np.std(k_eff)),
        "min_k_eff": float(np.min(k_eff)),
        "max_k_eff": float(np.max(k_eff)),
        "k_eff": k_eff,
    }

    logger.info(
        f"EffectiveNeighborhoodSize[{mode}]: mean={result['mean_k_eff']:.2f}, "
        f"median={result['median_k_eff']:.2f}, "
        f"std={result['std_k_eff']:.2f}"
    )
    return result
