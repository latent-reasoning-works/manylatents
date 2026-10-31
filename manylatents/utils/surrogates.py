"""Surrogate data: keep some structure of an array, destroy the rest.

Each function is a control for the question "would this score look the same if
structure X were absent?". All take a ``numpy.random.Generator`` and return new
arrays; inputs are never modified.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable


def _matrix(x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or 0 in x.shape:
        raise ValueError(f"expected a nonempty 2D array, got shape {x.shape}")
    return x


def shuffle_within_rows(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the values of every row independently.

    Keeps each row's multiset of values, so its max, norm and mean, and
    destroys which column each value sat in. Any score that survives this
    shuffle depended only on per-row magnitudes.
    """
    return rng.permuted(_matrix(x), axis=1)


def random_feature_subset(
    x: np.ndarray, n_features: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A random subset of columns, in their original order. Returns (subset, indices)."""
    x = _matrix(x)
    if (isinstance(n_features, (bool, np.bool_))
            or not isinstance(n_features, (int, np.integer))
            or not 0 < n_features <= x.shape[1]):
        raise ValueError(
            f"n_features must be an integer in 1..{x.shape[1]}, got {n_features}"
        )
    indices = np.sort(rng.choice(x.shape[1], size=int(n_features), replace=False))
    return x[:, indices], indices


def gaussian_surrogate(
    x: np.ndarray, rng: np.random.Generator, n_samples: Optional[int] = None
) -> np.ndarray:
    """Draw from the Gaussian with the mean and sample covariance of ``x``.

    Keeps first- and second-order structure and nothing else. Sampling goes
    through the thin SVD of the centred data, so no d-by-d covariance is formed
    and more features than rows is fine.
    """
    x = _matrix(x)
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise MeasurementUnavailable("x must be real numeric data")
    x = np.asarray(x, dtype=np.float64)
    if n_samples is not None and (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer)) or n_samples < 1
    ):
        raise MeasurementUnavailable("n_samples must be a positive nonboolean integer")
    n = x.shape[0]
    if n < 2:
        raise MeasurementUnavailable("a covariance needs at least two rows")
    if not np.isfinite(x).all():
        raise MeasurementUnavailable("x contains non-finite values")
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    draws = rng.standard_normal((n if n_samples is None else int(n_samples), singular.size))
    return mean + (draws * (singular / np.sqrt(n - 1))) @ vt


def permute_within_groups(
    values: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute ``values`` among the members of each group.

    The null for a group-stratified statistic: every group keeps its own
    values, their assignment to members is random.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)
    if values.ndim != 1 or values.shape != groups.shape:
        raise ValueError(
            f"values and groups must be 1D and the same length; got {values.shape} and {groups.shape}"
        )
    if np.any(groups != groups):
        raise MeasurementUnavailable("groups contain missing keys")
    out = values.copy()
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        out[members] = values[rng.permutation(members)]
    return out
