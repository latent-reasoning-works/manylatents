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
    resample: str = "rows",
) -> dict[str, Any]:
    """Group-size-weighted mean of per-group average precision.

    Args:
        scores: (n,) finite scores; higher means more likely positive.
        labels: (n,) binary labels (bool, or integers 0 and 1).
        groups: (n,) group key per row.
        n_bootstrap: integer bootstrap replicates, at least 2; 0 skips it.
        rng: generator for the bootstrap; required when ``n_bootstrap > 0``.
        resample: what a bootstrap replicate redraws. The two answer different
            questions and can differ severalfold.
            ``"rows"`` (default) resamples rows with replacement inside every
            (group, label) cell, so both classes stay present in every group:
            the uncertainty from having finitely many rows per group.
            ``"groups"`` resamples whole groups with replacement and keeps each
            group's value fixed: the uncertainty from having finitely many
            groups that differ from one another. Needs at least two groups.

    Returns:
        ``{"auprc", "se", "per_group", "weights"}``. ``se`` is None without a
        bootstrap.

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
    if resample not in ("rows", "groups"):
        raise ValueError(f"resample must be 'rows' or 'groups', got {resample!r}")
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
    if n_bootstrap and resample == "groups":
        keys = list(per_group)
        if len(keys) < 2:
            raise MeasurementUnavailable(
                "resampling groups needs at least two groups; got one"
            )
        values = np.array([per_group[g] for g in keys])
        sizes = np.array([weights[g] for g in keys], dtype=np.float64)
        replicates = np.empty(int(n_bootstrap), dtype=np.float64)
        for b in range(int(n_bootstrap)):
            draw = rng.integers(0, len(keys), size=len(keys))
            replicates[b] = (values[draw] * sizes[draw]).sum() / sizes[draw].sum()
        se = float(np.std(replicates, ddof=1))
    elif n_bootstrap:
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


def grouped_average_precision_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Paired difference of group-size-weighted average precisions (a minus b).

    Scores, binary labels and group keys are aligned 1D arrays, as in
    :func:`grouped_average_precision`. Higher scores mean more likely positive.
    ``n_bootstrap`` must be an integer >= 2 and ``rng`` a numpy Generator.

    Rows are drawn with replacement within every (group, label) cell. When
    ``clusters`` is supplied, a second bootstrap draws each group's clusters
    with replacement, retaining all rows of each drawn cluster. Cluster keys
    must be aligned with the rows, belong to exactly one group and contain
    both classes. Each replicate uses the same draw for both scores and
    weights group average precisions by their resampled row counts.

    Returns:
        ``auprc_a``, ``auprc_b``, ``difference``, ``ci95_rows``,
        ``ci95_clusters``, ``n_clusters``, ``groups_ahead`` and ``n_groups``.
        Intervals are lists of the 2.5% and 97.5% quantiles of replicate
        differences. Cluster outputs are None when clusters are absent.
        ``groups_ahead`` counts groups with strictly higher AP for a.

    Raises:
        MeasurementUnavailable: invalid bootstrap settings, unavailable input
            measurements, or invalid clusters (offending keys are named).
        ValueError: input arrays are not 1D and aligned.
    """
    if (isinstance(n_bootstrap, (bool, np.bool_))
            or not isinstance(n_bootstrap, (int, np.integer))
            or n_bootstrap < 2):
        raise MeasurementUnavailable("n_bootstrap must be an integer >= 2")
    if not isinstance(rng, np.random.Generator):
        raise MeasurementUnavailable("a bootstrap needs an explicit numpy Generator in rng")

    observed_a = grouped_average_precision(scores_a, labels, groups)
    observed_b = grouped_average_precision(scores_b, labels, groups)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    groups = np.asarray(groups)
    group_rows = {g: np.flatnonzero(groups == g) for g in observed_a["per_group"]}

    cluster_members = None
    n_clusters = None
    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.shape != labels.shape:
            raise ValueError("clusters must be 1D and the same length as labels")
        missing = (clusters != clusters) | (clusters == None)  # noqa: E711
        if np.any(missing):
            raise unavailable_for_points(
                f"clusters contain missing keys: {clusters[missing].tolist()}", missing
            )
        cluster_rows = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters).tolist()}
        spanning = [c for c, rows in cluster_rows.items() if np.unique(groups[rows]).size != 1]
        one_class = [c for c, rows in cluster_rows.items() if np.unique(labels[rows]).size != 2]
        if spanning or one_class:
            raise MeasurementUnavailable(
                f"clusters spanning groups: {spanning}; clusters without both classes: {one_class}"
            )
        cluster_members = {
            g: [cluster_rows[c] for c in np.unique(clusters[rows]).tolist()]
            for g, rows in group_rows.items()
        }
        n_clusters = len(cluster_rows)

    cells = {
        g: [rows[labels[rows] == cls] for cls in (0, 1)]
        for g, rows in group_rows.items()
    }
    intervals = {}
    for mode in ("rows", "clusters"):
        if mode == "clusters" and cluster_members is None:
            intervals[mode] = None
            continue
        replicates = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            if mode == "rows":
                sampled_rows = {
                    g: np.concatenate([rng.choice(cell, size=cell.size, replace=True) for cell in pair])
                    for g, pair in cells.items()
                }
            else:
                sampled_rows = {
                    g: np.concatenate([
                        members[j] for j in rng.integers(0, len(members), size=len(members))
                    ])
                    for g, members in cluster_members.items()
                }
            a, _, _ = _statistic(scores_a, labels, sampled_rows)
            b_value, _, _ = _statistic(scores_b, labels, sampled_rows)
            replicates[b] = a - b_value
        intervals[mode] = np.quantile(replicates, [0.025, 0.975]).tolist()

    return {
        "auprc_a": observed_a["auprc"],
        "auprc_b": observed_b["auprc"],
        "difference": observed_a["auprc"] - observed_b["auprc"],
        "ci95_rows": intervals["rows"],
        "ci95_clusters": intervals["clusters"],
        "n_clusters": n_clusters,
        "groups_ahead": sum(
            observed_a["per_group"][g] > observed_b["per_group"][g] for g in group_rows
        ),
        "n_groups": len(group_rows),
    }
