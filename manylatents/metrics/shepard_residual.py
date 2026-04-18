"""Per-point Shepard residual on local-rank neighborhoods.

For each point :math:`i`, the metric measures how much each ambient
distance ``d_amb(i, j)`` (over :math:`i`'s :math:`k`-NN in the ambient
space) deviates from a single global linear fit ``d_emb = alpha * d_amb``
through the embedding distances. This gives a per-point "the embedding
moved this point farther from its ambient neighbors than a uniform stretch
would predict" residual — the ambient-space anchor that pairs with the
mismatch ratio :math:`v`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.knn import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["shepard_residual", "shepard"],
    default_params={"k": 15},
    description="Per-point Shepard residual on local-rank neighborhoods",
)
def ShepardResidual(
    embeddings: np.ndarray,
    dataset=None,
    module=None,
    k: int = 15,
    cache: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute per-point Shepard residual on local-rank neighborhoods.

    For each point :math:`i`, look up the :math:`k` ambient nearest
    neighbors :math:`\\mathcal{N}_i`. Collect the corresponding pairs
    ``(d_amb_ij, d_emb_ij)`` for ``j`` in :math:`\\mathcal{N}_i`. A single
    global slope :math:`\\alpha` is fit through the origin across all such
    pairs (ordinary least-squares, no intercept). The per-point residual is
    the mean absolute deviation from the fit over :math:`i`'s neighborhood::

        r_i = mean_{j in N_i} |d_emb_ij - alpha * d_amb_ij|

    Requires both a ``dataset`` (with ``.data``) for the ambient-space
    distances and the ``embeddings`` for the embedded distances. The
    ``module`` argument is unused.

    Args:
        embeddings: ``(n, d_emb)`` embedded coordinates.
        dataset: Object with ``.data`` ``(n, d_amb)`` ambient coordinates.
        module: Unused (accepted for protocol compatibility).
        k: Local neighborhood size.
        cache: Shared kNN cache.

    Returns:
        Dict with:
            residual: ``(n,)`` per-point Shepard residual.
            alpha: Global slope from the through-origin OLS fit.
            mean_residual, median_residual, std_residual: summary stats.
            mean_residual_normalized: ``mean_residual / alpha`` — scale-free
                summary in units of "ambient distance".
    """
    if cache is None:
        cache = {}

    if dataset is None or not hasattr(dataset, "data"):
        raise ValueError(
            "ShepardResidual requires `dataset` with a `.data` attribute "
            "(ambient-space coordinates)."
        )

    ambient = np.asarray(dataset.data, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_points = ambient.shape[0]
    if embeddings.shape[0] != n_points:
        raise ValueError(
            f"ShepardResidual: embeddings has {embeddings.shape[0]} rows but "
            f"dataset.data has {n_points} rows."
        )

    # Ambient-space kNN. compute_knn(k=k, include_self=True) returns
    # (n, k+1) with self at column 0 and k actual neighbors at columns 1..k.
    d_amb_full, idx_amb = compute_knn(ambient, k=k, include_self=True, cache=cache)
    d_amb = d_amb_full[:, 1:]  # (n, k)
    nbr_idx = idx_amb[:, 1:]    # (n, k)

    # Corresponding embedded distances for the same (i, j) pairs
    d_emb = np.linalg.norm(
        embeddings[:, None, :] - embeddings[nbr_idx],
        axis=2,
    )  # (n, k)

    flat_amb = d_amb.reshape(-1)
    flat_emb = d_emb.reshape(-1)

    # OLS slope through origin: alpha = sum(x*y) / sum(x^2)
    denom = float((flat_amb ** 2).sum())
    if denom <= 0:
        logger.warning("ShepardResidual: zero variance in ambient distances; alpha=1.0")
        alpha = 1.0
    else:
        alpha = float((flat_amb * flat_emb).sum() / denom)

    residual_per_pair = np.abs(d_emb - alpha * d_amb)  # (n, k)
    residual = residual_per_pair.mean(axis=1)          # (n,)

    mean_r = float(residual.mean())
    summary = {
        "residual": residual,
        "alpha": alpha,
        "mean_residual": mean_r,
        "median_residual": float(np.median(residual)),
        "std_residual": float(residual.std()),
        "mean_residual_normalized": mean_r / alpha if alpha != 0 else float("nan"),
    }

    logger.info(
        f"ShepardResidual: alpha={alpha:.3f}, mean_residual={mean_r:.3f}, "
        f"median_residual={summary['median_residual']:.3f}"
    )
    return summary
