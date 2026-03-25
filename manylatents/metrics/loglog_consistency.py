import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["loglog_consistency", "power_law_consistency", "lid_reliability"],
    default_params={"k": 200, "k_min": 5, "k_steps": 20},
    description="Per-point log-log power law consistency of kNN distance scaling",
)
def LogLogConsistency(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: int = 200,
    k_min: int = 5,
    k_steps: int = 20,
    cache: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Per-point log-log power law consistency diagnostic.

    For each point, sweeps k from k_min to k (log-spaced) and fits
    log(k) = m * log(d_k) + c. The slope m directly estimates the local
    intrinsic dimension. R^2 measures whether the power law k ~ d_k^m
    holds locally.

    Args:
        embeddings: (n_samples, n_features) array.
        dataset: Unused, kept for protocol compatibility.
        module: Unused, kept for protocol compatibility.
        k: Maximum number of neighbors (drives kNN pre-computation).
        k_min: Minimum k for the sweep.
        k_steps: Number of log-spaced k values between k_min and k.
        cache: Optional shared cache dict for kNN reuse.

    Returns:
        Dict with summary scalars and per-point arrays.
    """
    distances, _ = compute_knn(embeddings, k=k, include_self=True, cache=cache)

    # Clamp k to actual columns returned (compute_knn may reduce k for small datasets)
    k = distances.shape[1] - 1  # last valid index (column 0 = self)

    # Build log-spaced k grid
    k_values = np.unique(
        np.logspace(np.log10(k_min), np.log10(k), k_steps).astype(int)
    )

    # Slice distance columns: with include_self=True, column j = j-th neighbor
    T = distances[:, k_values]  # (n_points, len(k_values))

    # Guard against T_k = 0 (duplicate points)
    eps = 1e-30
    log_T = np.log(np.maximum(T, eps))  # (n_points, len(k_values))
    log_k = np.log(k_values.astype(float))  # (len(k_values),)

    # Vectorized linear regression: log_k = slope * log_T + intercept
    # x = log_T (distance, geometric variable), y = log_k (count)
    # slope = m (intrinsic dimension) directly
    n_k = len(k_values)
    sum_x = log_T.sum(axis=1)            # (n_points,)
    sum_x2 = (log_T ** 2).sum(axis=1)    # (n_points,)
    sum_y = log_k.sum()                   # scalar (same for all points)
    sum_xy = (log_T * log_k).sum(axis=1)  # (n_points,)

    denom = n_k * sum_x2 - sum_x ** 2
    slope = np.where(np.abs(denom) > 1e-20,
                     (n_k * sum_xy - sum_x * sum_y) / denom, 0.0)
    intercept = (sum_y - slope * sum_x) / n_k

    # R^2 = 1 - SS_res / SS_tot
    y_pred = slope[:, None] * log_T + intercept[:, None]  # (n_points, n_k)
    ss_res = ((log_k - y_pred) ** 2).sum(axis=1)
    ss_tot = ((log_k - log_k.mean()) ** 2).sum()  # scalar, same for all
    r_squared = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    r_squared = np.clip(r_squared, 0.0, 1.0)

    # Population-level summary: mean log(d_k) at each k (for log-log plot)
    mean_log_T = log_T.mean(axis=0)  # (len(k_values),)

    result = {
        "mean_r_squared": float(np.mean(r_squared)),
        "std_r_squared": float(np.std(r_squared)),
        "frac_reliable": float(np.mean(r_squared > 0.95)),
        "mean_slope": float(np.mean(slope)),
        "slope": slope,
        "r_squared": r_squared,
        "k_values": k_values,
        "mean_log_T": mean_log_T,
    }

    logger.info(
        f"LogLogConsistency: mean_R²={result['mean_r_squared']:.3f}, "
        f"frac_reliable={result['frac_reliable']:.3f}, "
        f"mean_slope(m)={result['mean_slope']:.3f}"
    )
    return result
