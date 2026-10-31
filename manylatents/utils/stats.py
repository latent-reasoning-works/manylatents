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


def partial_spearman(x, y, covariates) -> dict:
    """Spearman correlation of ``x`` and ``y`` after removing covariates.

    All variables are rank-transformed (average ranks for ties). The ranks of
    ``x`` and ``y`` are each regressed on an intercept and the covariate ranks;
    ``rho`` is the Pearson correlation of the residuals. With no covariates
    this is the ordinary Spearman correlation.

    Args:
        x, y: (n,) arrays.
        covariates: (n,) or (n, q) array; q may be 0.

    Returns:
        ``{"rho", "p_value", "n", "dof"}`` with ``dof = n - 2 - q`` and a
        two-sided p-value from the t distribution.

    Raises:
        MeasurementUnavailable: non-finite input, fewer than one residual
            degree of freedom, or a variable fully explained by the covariates.
    """
    from scipy.stats import rankdata, t as t_distribution

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(covariates, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if x.ndim != 1 or y.shape != x.shape or z.ndim != 2 or z.shape[0] != x.shape[0]:
        raise ValueError(
            f"x and y must be (n,), covariates (n,) or (n, q); got {x.shape}, {y.shape}, {z.shape}"
        )
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise MeasurementUnavailable("partial Spearman needs finite inputs")
    n, q = z.shape
    dof = n - 2 - q
    if dof < 1:
        raise MeasurementUnavailable(
            f"n={n} leaves {dof} residual degrees of freedom with {q} covariates"
        )

    design = np.column_stack([np.ones(n)] + [rankdata(z[:, j]) for j in range(q)])

    def residual(values):
        ranks = rankdata(values)
        coefficients, *_ = np.linalg.lstsq(design, ranks, rcond=None)
        return ranks - design @ coefficients, ranks

    rx, ranks_x = residual(x)
    ry, ranks_y = residual(y)
    # A residual that is zero up to rounding means the covariates explain the variable.
    for name, res, ranks in (("x", rx, ranks_x), ("y", ry, ranks_y)):
        spread = np.linalg.norm(ranks - ranks.mean())
        if spread == 0 or np.linalg.norm(res) <= 1e-9 * spread:
            raise MeasurementUnavailable(
                f"{name} has no variation left after removing the covariates"
            )
    rho = float(np.clip(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry)), -1.0, 1.0))
    if abs(rho) == 1.0:
        p_value = 0.0
    else:
        statistic = rho * np.sqrt(dof / (1.0 - rho * rho))
        p_value = float(2.0 * t_distribution.sf(abs(statistic), dof))
    return {"rho": rho, "p_value": p_value, "n": int(n), "dof": int(dof)}
