"""Per-point mismatch ratio v = k_eff / k_star.

Compares the effective neighborhood each DR method uses (k_eff, from its
internal affinity matrix) against the scale where the input geometry is
Poisson-valid (k_star, from log-log consistency on the input space).

Overshoot (v > 1) corrupts distances; undershoot (v << 1) severs connections.
"""
import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.knn import compute_knn

logger = logging.getLogger(__name__)


def _compute_kstar(
    data: np.ndarray,
    k_max: int = 200,
    k_min: int = 5,
    k_steps: int = 20,
    r2_threshold: float = 0.95,
    cache: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point k_star: largest k where cumulative R² > threshold.

    Vectorized over all points (no per-point loop).

    Args:
        data: (n, d) input-space coordinates (e.g. PCA-50).
        k_max: Maximum k for the log-log sweep.
        k_min: Minimum k.
        k_steps: Number of log-spaced k values.
        r2_threshold: R² threshold for valid geometric regime.
        cache: Shared kNN cache.

    Returns:
        k_star: (n,) per-point regime boundary.
        k_values: log-spaced k sweep points used.
    """
    if cache is None:
        cache = {}

    distances, _ = compute_knn(data, k=k_max, include_self=True, cache=cache)

    k_values = np.unique(
        np.logspace(np.log10(k_min), np.log10(k_max), k_steps).astype(int)
    )
    max_col = distances.shape[1] - 1
    k_values = k_values[k_values <= max_col]

    n_points = data.shape[0]
    k_star = np.full(n_points, float(k_values[0]))

    # Sweep from largest k down, find where R² > threshold (vectorized)
    for j in range(len(k_values), 2, -1):
        sub_k = k_values[:j]
        T = distances[:, sub_k]
        eps = 1e-30
        log_T = np.log(np.maximum(T, eps))
        log_k = np.log(sub_k.astype(float))
        n_k = j
        sx = log_k.sum()
        sx2 = (log_k ** 2).sum()
        sy = log_T.sum(axis=1)
        sxy = (log_T * log_k).sum(axis=1)
        d = n_k * sx2 - sx ** 2
        sl = (n_k * sxy - sx * sy) / d
        ic = (sy - sl * sx) / n_k
        yp = sl[:, None] * log_k + ic[:, None]
        ss_res = ((log_T - yp) ** 2).sum(axis=1)
        ym = log_T.mean(axis=1, keepdims=True)
        ss_tot = ((log_T - ym) ** 2).sum(axis=1)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
        r2 = np.clip(r2, 0, 1)
        mask = (r2 >= r2_threshold) & (k_star <= k_values[0])
        k_star[mask] = float(sub_k[-1])

    return k_star, k_values


def _compute_keff(module) -> np.ndarray:
    """Extract per-point k_eff from a fitted module's affinity matrix."""
    W = module.affinity(ignore_diagonal=True, use_symmetric=False)
    if hasattr(W, 'toarray'):
        W = W.toarray()
    W = np.asarray(W)
    row_sum = W.sum(axis=1)
    row_sum_sq = (W ** 2).sum(axis=1)
    return np.where(row_sum_sq > 0, row_sum ** 2 / row_sum_sq, 0.0)


@register_metric(
    aliases=["mismatch_ratio", "v_ratio", "mismatch"],
    default_params={"k": 200, "k_min": 5, "k_steps": 20, "r2_threshold": 0.95},
    description="Per-point mismatch ratio v = k_eff / k_star",
)
def MismatchRatio(
    embeddings: np.ndarray,
    dataset=None,
    module=None,
    k: int = 200,
    k_min: int = 5,
    k_steps: int = 20,
    r2_threshold: float = 0.95,
    cache: Optional[dict] = None,
    **kwargs,
) -> dict[str, Any]:
    """Compute per-point mismatch ratio v = k_eff / k_star.

    k_star is computed on the **dataset** (input space, e.g. PCA-50).
    k_eff is computed from the **module**'s internal affinity matrix.

    Requires both a dataset (with .data attribute) and a fitted module.

    Returns dict with:
        k_star: (n,) per-point Poisson-valid scale
        k_eff: (n,) per-point effective neighborhood size
        v: (n,) mismatch ratio
        mean_v, median_v, std_v: summary stats
        frac_overshoot: fraction with v > 1
        frac_undershoot: fraction with v < 1
        mean_k_star, median_k_star: summary stats
    """
    if cache is None:
        cache = {}

    # Get input-space data for k_star computation
    input_data = None
    if dataset is not None and hasattr(dataset, 'data'):
        input_data = dataset.data
    if input_data is None:
        logger.warning(
            "MismatchRatio: no dataset.data available for k_star computation. "
            "Falling back to computing k_star on the embedding itself."
        )
        input_data = embeddings

    input_data = np.asarray(input_data, dtype=np.float32)
    n_points = input_data.shape[0]

    # k_star from input space
    k_star, k_values = _compute_kstar(
        input_data, k_max=k, k_min=k_min, k_steps=k_steps,
        r2_threshold=r2_threshold, cache=cache,
    )

    # k_eff from module's affinity matrix
    if module is not None:
        try:
            k_eff = _compute_keff(module)
            if len(k_eff) != n_points:
                logger.warning(
                    f"MismatchRatio: k_eff length {len(k_eff)} != n_points {n_points} "
                    f"(likely landmarks). Using mean k_eff={k_eff.mean():.1f}."
                )
                k_eff = np.full(n_points, k_eff.mean())
        except (NotImplementedError, AttributeError) as e:
            ns = getattr(module, 'neighborhood_size', None) or 15
            logger.warning(f"MismatchRatio: affinity unavailable ({e}). Using uniform k_eff={ns}.")
            k_eff = np.full(n_points, float(ns))
    else:
        logger.warning("MismatchRatio: no module provided. Cannot compute k_eff.")
        k_eff = np.full(n_points, np.nan)

    # v = k_eff / k_star
    v = np.where(k_star > 0, k_eff / k_star, 0.0)

    logger.info(
        f"MismatchRatio: median v={np.median(v):.3f}, "
        f"overshoot={100 * (v > 1).mean():.1f}%, "
        f"undershoot={100 * (v < 1).mean():.1f}%"
    )

    return {
        "k_star": k_star,
        "k_eff": k_eff,
        "v": v,
        "mean_v": float(np.mean(v)),
        "median_v": float(np.median(v)),
        "std_v": float(np.std(v)),
        "frac_overshoot": float((v > 1).mean()),
        "frac_undershoot": float((v < 1).mean()),
        "mean_k_star": float(np.mean(k_star)),
        "median_k_star": float(np.median(k_star)),
    }
