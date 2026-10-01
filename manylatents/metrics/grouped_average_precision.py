"""Average precision computed within groups and averaged by group size.

For ranking tasks where rows fall into groups that should not be pooled into
one ranking (held-out blocks, strata, batches). A library function, not a
registered metric: it needs labels and groups, which the metric protocol has
no slot for.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import average_precision_score

from manylatents.utils.exceptions import MeasurementUnavailable, unavailable_for_points


def _statistic(scores, labels, group_rows) -> tuple[float, dict[Any, float], dict[Any, int]]:
    per_group, weights = {}, {}
    for group, rows in group_rows.items():
        per_group[group] = float(average_precision_score(labels[rows], scores[rows]))
        weights[group] = int(rows.size)
    total = sum(weights.values())
    value = sum(per_group[g] * weights[g] for g in per_group) / total
    return float(value), per_group, weights


def grouped_average_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Group-size-weighted mean of per-group average precision.

    Args:
        scores: (n,) finite scores; higher means more likely positive.
        labels: (n,) binary labels (bool, or integers 0 and 1).
        groups: (n,) group key per row.
        n_bootstrap: integer bootstrap replicates, at least 2; 0 skips it.
        rng: generator for the bootstrap; required when ``n_bootstrap > 0``.

    Returns:
        ``{"auprc", "se", "per_group", "weights"}``. ``se`` is None without a
        bootstrap. Each replicate resamples rows with replacement inside every
        (group, label) cell, so both classes stay present in every group.

    Raises:
        MeasurementUnavailable: non-finite scores, non-binary labels, or a
            group without both classes. Such a group is named, not skipped:
            dropping it would change what the weighted mean averages over.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != groups.shape:
        raise ValueError(
            "scores, labels and groups must be 1D and the same length; got "
            f"{scores.shape}, {labels.shape}, {groups.shape}"
        )
    if scores.size == 0:
        raise MeasurementUnavailable("no rows to score")
    if not np.isfinite(scores).all():
        raise unavailable_for_points("scores are not finite", ~np.isfinite(scores))
    if labels.dtype != bool:
        if not np.isin(labels, (0, 1)).all():
            raise MeasurementUnavailable("labels must be binary (bool, or 0 and 1)")
    labels = labels.astype(int)
    if (isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer))
            or n_bootstrap < 0 or n_bootstrap == 1):
        raise MeasurementUnavailable("n_bootstrap must be 0 or an integer >= 2")
    if n_bootstrap and not isinstance(rng, np.random.Generator):
        raise ValueError("a bootstrap needs an explicit numpy Generator in rng")
    if np.any(groups != groups):
        raise unavailable_for_points("groups contain missing keys", groups != groups)

    group_rows = {g: np.flatnonzero(groups == g) for g in np.unique(groups).tolist()}
    one_class = [
        g for g, rows in group_rows.items() if np.unique(labels[rows]).size < 2
    ]
    if one_class:
        raise MeasurementUnavailable(
            f"average precision is undefined in groups without both classes: {one_class}"
        )

    value, per_group, weights = _statistic(scores, labels, group_rows)

    se = None
    if n_bootstrap:
        cells = [
            rows[labels[rows] == cls]
            for rows in group_rows.values()
            for cls in (0, 1)
        ]
        replicates = np.empty(int(n_bootstrap), dtype=np.float64)
        for b in range(int(n_bootstrap)):
            sample = np.concatenate(
                [rng.choice(cell, size=cell.size, replace=True) for cell in cells]
            )
            sampled_groups = groups[sample]
            sampled_rows = {
                g: np.flatnonzero(sampled_groups == g) for g in group_rows
            }
            replicates[b], _, _ = _statistic(scores[sample], labels[sample], sampled_rows)
        se = float(np.std(replicates, ddof=1))

    return {"auprc": value, "se": se, "per_group": per_group, "weights": weights}
