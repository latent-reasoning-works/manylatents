"""Dimensionless shape statistics on an exact float64 parent graph."""
from importlib.metadata import version

import numpy as np

from manylatents.metrics.lid import _distinct_rows, _lid_from_distances, _rms_normalise
from manylatents.utils.exceptions import MeasurementUnavailable

GRID = (5, 10, 20, 40, 80, 160)


def _exact_knn(points, k, *, chunk_size=256):
    """Return sorted distances and indices, breaking ties by original index."""
    from scipy.spatial.distance import cdist
    n = len(points)
    distances = np.empty((n, k), dtype=np.float64)
    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = cdist(points[start:stop], points)
        block[np.arange(stop-start), np.arange(start, stop)] = np.inf
        order = np.argsort(block, axis=1, kind="stable")[:, :k]
        indices[start:stop] = order
        distances[start:stop] = np.take_along_axis(block, order, axis=1)
    return distances, indices


def _window(medians, usable):
    """Find the longest stable consecutive grid interval, preferring smaller k."""
    best = None
    length = 0
    for i, lo in enumerate(GRID):
        values = []
        for j in range(i, len(GRID)):
            k = GRID[j]
            if k not in usable or k not in medians:
                break
            values.append(medians[k])
            if min(values) <= 0 or not np.isfinite(values).all():
                break
            if max(values) / min(values) > 1.25:
                break
            if len(values) >= 2 and len(values) > length:
                best, length = (lo, k), len(values)
    return best


def _row(kind, name, k, aggregation, n, value=None, status="ok", reason=""):
    return dict(entity_kind=kind, stat_name=name, scale_kind="none" if k is None else "k",
                scale_value=k, aggregation=aggregation,
                value=None if value is None else float(value), n_cells_used=int(n),
                status=status, reason=reason, knn_backend="scipy_exact", dtype="float64",
                metric_version=f"manylatents=={version('manylatents')}:shape_profile")


def _scale_failure(k, n):
    if k >= n:
        return "insufficient_neighbors"
    if n < 10*k:
        return "too_few_points"
    return ""


def _core_rows(logs, lids, aggregate, *, kind, n_cells, n_distinct, failure):
    n = len(aggregate)
    small = kind == "selection" and n < 20
    rows, medians = [], {}
    usable = tuple(k for k in GRID if not _scale_failure(k, n_distinct))
    for k in GRID:
        reason = "too_few_selected" if small else _scale_failure(k, n_distinct)
        status = "undefined" if small else "censored"
        if not reason:
            reason, status = failure, "undefined"
        if not reason and k not in logs:
            reason, status = "insufficient_reference_neighbors", "censored"
        vals = logs[k][aggregate] if not reason else None
        if vals is not None and not np.isfinite(vals).all():
            reason, status = "precision_loss", "undefined"
        for aggregation in ("median", "iqr"):
            value = None
            if not reason:
                value = np.median(vals) if aggregation == "median" else np.diff(np.quantile(vals, [.25, .75]))[0]
            rows.append(_row(kind, "log_rk_r15", k, aggregation, n, value,
                             status if reason else "ok", reason))
        if not small and not failure and k in lids:
            estimates = lids[k][aggregate]
            if len(estimates) and np.isfinite(estimates).all():
                medians[k] = float(np.median(estimates))
    window = _window(medians, usable)
    for name, endpoint in (("window_lo", 0), ("window_hi", 1)):
        if small or failure:
            r = _row(kind, name, None, "value", n, status="undefined",
                     reason="too_few_selected" if small else failure)
        elif window is None:
            r = _row(kind, name, None, "value", n, status="empty", reason="no_stable_window")
        else:
            censored = endpoint == 1 and window[1] == max(usable)
            r = _row(kind, name, None, "value", n, window[endpoint],
                     "censored" if censored else "ok", "window_right_censored" if censored else "")
        rows.append(r)
    rows.append(_row(kind, "n_cells", None, "value", n, n_cells))
    return rows


def shape_profile(points, *, selection=None, layout=None, extended=False):
    """Compute fixed dataset rows and optional selection rows from parent geometry."""
    if type(extended) is not bool:
        raise ValueError("extended must be boolean")
    if extended:
        raise ValueError("not_implemented")
    x = np.asarray(points)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("points must be a real numeric matrix")
    if selection is not None:
        selection = np.asarray(selection)
        if selection.dtype != bool or selection.shape != (len(x),):
            raise ValueError("selection must be a boolean mask of length n")
    if layout is not None:
        layout = np.asarray(layout)
        if layout.shape != (len(x), 2) or not np.issubdtype(layout.dtype, np.number) or np.iscomplexobj(layout):
            raise ValueError("layout must be a real n by 2 matrix")
    failure = ""
    if 0 in x.shape:
        unique, inverse = x, np.arange(len(x))
        failure = "normalisation_unresolved"
    else:
        unique, inverse = _distinct_rows(x)
        # Restore first-occurrence order after the shared exact deduplication.
        first = np.full(len(unique), len(x), dtype=int)
        np.minimum.at(first, inverse, np.arange(len(x)))
        order = np.argsort(first)
        remap = np.empty(len(order), dtype=int)
        remap[order] = np.arange(len(order))
        unique, inverse = unique[order], remap[inverse]
    n = len(unique)
    conditioned = None
    if not np.isfinite(x).all():
        failure = "nonfinite_input"
    if not failure:
        try:
            conditioned = _rms_normalise(unique)
            if len(_distinct_rows(conditioned)[0]) != n:
                failure = "precision_loss"
        except MeasurementUnavailable:
            failure = "normalisation_unresolved"
    usable = [k for k in GRID if not _scale_failure(k, n)]
    logs, lids = {}, {}
    indices = None
    if not failure and usable:
        distances, indices = _exact_knn(conditioned, max(15, max(usable)))
        for k in usable:
            with np.errstate(divide="ignore", invalid="ignore"):
                logs[k] = np.log(distances[:, k-1] / distances[:, 14])
            estimates = np.full(n, np.nan)
            for i in range(n):
                try:
                    estimates[i] = _lid_from_distances(distances[i:i+1, :k])[0]
                except MeasurementUnavailable:
                    pass
            lids[k] = estimates
    dataset = _core_rows(logs, lids, np.arange(n), kind="dataset", n_cells=len(x),
                         n_distinct=n, failure=failure)
    if selection is None:
        return dataset
    aggregate = np.unique(inverse[selection])
    selected = _core_rows(logs, lids, aggregate, kind="selection",
                          n_cells=int(selection.sum()), n_distinct=n, failure=failure)
    edges, layout_edges = None, None
    if usable and not failure:
        edges = indices if n == len(x) else _exact_knn(conditioned[inverse], max(usable))[1]
        if layout is not None and np.isfinite(layout).all():
            try:
                layout_edges = _exact_knn(_rms_normalise(layout), max(usable))[1]
            except MeasurementUnavailable:
                # Constant layouts still have a well-defined index-tied graph.
                layout_edges = _exact_knn(layout, max(usable))[1]
    count = int(selection.sum())
    for name in ("conductance", "selection_fidelity"):
        for k in GRID:
            reason = _scale_failure(k, n)
            status = "censored" if reason else "undefined"
            value = None
            if not count:
                reason, status = "empty_selection", "undefined"
            elif not reason:
                reason = failure
                if not reason:
                    retention = float(selection[edges[selection, :k]].mean())
                    if name == "conductance":
                        value = 1. - retention
                    elif layout_edges is None:
                        reason = "layout_unavailable"
                    else:
                        denominator = float(selection[layout_edges[selection, :k]].mean())
                        if denominator == 0:
                            reason = "zero_layout_retention"
                        else:
                            value = retention / denominator
            selected.append(_row("selection", name, k, "share" if name == "conductance" else "value",
                                 count, value, status if reason else "ok", reason))
    priority = {"ok": 0, "censored": 1, "empty": 2, "undefined": 3}
    for parent, child in zip(dataset[:14], selected[:14]):
        winner = child if priority[child["status"]] >= priority[parent["status"]] else parent
        side = "selection" if winner is child else "dataset"
        ok = winner["status"] == "ok"
        selected.append(_row("selection", "relative_"+child["stat_name"], child["scale_value"],
                             child["aggregation"], len(aggregate),
                             child["value"] - parent["value"] if ok else None,
                             winner["status"], "" if ok else f"relative_{side}_{winner['reason']}"))
    return dataset + selected
