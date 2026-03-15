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

    For each point, sweeps k from k_min to k (log-spaced) and measures whether
    the k-th neighbor distance T_k grows as k^(1/m). The R^2 of the log-log
    linear fit is a per-point reliability score for LID estimation.

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

    # Vectorized linear regression: log_T = slope * log_k + intercept
    n_k = len(k_values)
    sum_x = log_k.sum()
    sum_x2 = (log_k ** 2).sum()
    sum_y = log_T.sum(axis=1)
    sum_xy = (log_T * log_k).sum(axis=1)

    denom = n_k * sum_x2 - sum_x ** 2
    slope = (n_k * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_k

    # R^2 = 1 - SS_res / SS_tot
    y_pred = slope[:, None] * log_k + intercept[:, None]
    ss_res = ((log_T - y_pred) ** 2).sum(axis=1)
    y_mean = log_T.mean(axis=1, keepdims=True)
    ss_tot = ((log_T - y_mean) ** 2).sum(axis=1)
    r_squared = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    r_squared = np.clip(r_squared, 0.0, 1.0)

    # LID from slope: slope = 1/m -> m = 1/slope
    lid_from_slope = np.where(np.abs(slope) > 1e-10, 1.0 / slope, 0.0)

    # Population-level summary
    mean_log_R = log_T.mean(axis=0)

    result = {
        "mean_r_squared": float(np.mean(r_squared)),
        "std_r_squared": float(np.std(r_squared)),
        "frac_reliable": float(np.mean(r_squared > 0.95)),
        "mean_slope": float(np.mean(slope)),
        "slope": slope,
        "lid_from_slope": lid_from_slope,
        "r_squared": r_squared,
        "k_values": k_values,
        "mean_log_R": mean_log_R,
    }

    logger.info(
        f"LogLogConsistency: mean_R²={result['mean_r_squared']:.3f}, "
        f"frac_reliable={result['frac_reliable']:.3f}, "
        f"mean_slope={result['mean_slope']:.3f}"
    )
    return result
