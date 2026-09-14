"""Statistical utilities for confidence intervals and resampling.

Provides a generic bootstrap CI function that wraps any statistic:

    >>> from manylatents.utils.stats import bootstrap_ci
    >>> ci = bootstrap_ci(lambda y, s: roc_auc_score(y, s), labels, scores)
    >>> print(ci)  # (0.731, 0.745)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable


def bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for any statistic on parallel arrays.

    Resamples rows (with replacement) from all arrays in lockstep,
    computes ``stat_fn`` on each resample, and returns percentile CI.

    Args:
        stat_fn: Callable that takes the same number of arrays as ``*arrays``
            and returns a scalar float. Every requested resample must succeed
            and return a finite value; otherwise the interval is unavailable.
        *arrays: One or more arrays of the same length (first axis).
        n_bootstrap: Number of bootstrap resamples (at least 10).
        ci: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` — the percentile confidence interval.

    Example::

        from sklearn.metrics import roc_auc_score
        lo, hi = bootstrap_ci(roc_auc_score, y_true, y_score, n_bootstrap=1000)
    """
    if not arrays:
        raise ValueError("At least one array is required")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must have the same length")

    if n == 0 or not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 10:
        raise MeasurementUnavailable("Bootstrap requires nonempty arrays and at least 10 resamples")
    if not 0 < ci < 1:
        raise MeasurementUnavailable("Bootstrap confidence level must lie strictly between 0 and 1")

    rng = np.random.RandomState(seed)
    stats: list[float] = []

    first_failure = None
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        resampled = tuple(a[idx] for a in arrays)
        try:
            val = np.asarray(stat_fn(*resampled))
            if val.ndim != 0 or not np.isfinite(val):
                raise ValueError("stat_fn did not return a finite scalar")
            stats.append(float(val))
        except Exception as exc:
            if first_failure is None:
                first_failure = exc
            continue

    if len(stats) != n_bootstrap:
        raise MeasurementUnavailable(
            f"Only {len(stats)}/{n_bootstrap} bootstrap resamples produced "
            f"finite values; all requested resamples must succeed."
        ) from first_failure

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(stats, 100 * alpha)),
        float(np.percentile(stats, 100 * (1 - alpha))),
    )
