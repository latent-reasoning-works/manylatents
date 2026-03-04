"""Statistical utilities for confidence intervals and resampling.

Provides a generic bootstrap CI function that wraps any statistic:

    >>> from manylatents.utils.stats import bootstrap_ci
    >>> ci = bootstrap_ci(lambda y, s: roc_auc_score(y, s), labels, scores)
    >>> print(ci)  # (0.731, 0.745)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


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
            and returns a scalar float.  Resamples that raise or return NaN
            are silently skipped.
        *arrays: One or more arrays of the same length (first axis).
        n_bootstrap: Number of bootstrap resamples.
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

    rng = np.random.RandomState(seed)
    stats: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        resampled = tuple(a[idx] for a in arrays)
        try:
            val = stat_fn(*resampled)
        except Exception:
            continue
        if np.isfinite(val):
            stats.append(val)

    if len(stats) < 10:
        raise RuntimeError(
            f"Only {len(stats)}/{n_bootstrap} bootstrap resamples produced "
            f"finite values — stat_fn may be failing on resampled data."
        )

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(stats, 100 * alpha)),
        float(np.percentile(stats, 100 * (1 - alpha))),
    )
