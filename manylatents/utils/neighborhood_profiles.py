"""Float64 measurements of balls, explicit patches, and nonnegative influence.

These raw instruments do not estimate intrinsic dimension or choose a scale.
Missing numeric entries are NaN with a corresponding status, so JSON consumers
must replace them with null. IDs always refer to input rows, including duplicates.
No approximate-neighbor cache is used.
"""
from __future__ import annotations

import hashlib
import numbers
import time
import numpy as np


def _matrix(value, name):
    array = np.asarray(value)
    if array.dtype != np.float64 or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite float64 matrix")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one coordinate")
    return array


def _radii(value):
    array = np.asarray(value)
    if (array.dtype != np.float64 or array.ndim != 1 or not len(array)
            or not np.isfinite(array).all() or np.any(array <= 0)
            or np.any(np.diff(array) <= 0)):
        raise ValueError("radii must be positive finite increasing float64 values")
    return array


def _ids(values, n, *, allow_none=False):
    result = []
    for value in values:
        if value is None and allow_none:
            result.append(None)
        elif (not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_))
              or not 0 <= value < n):
            raise ValueError("row identities must be integers in range")
        else:
            result.append(int(value))
    return result


def _frames(frame, p, dimension):
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.dtype != np.float64 or not np.isfinite(array).all():
        raise ValueError("frame must be finite float64")
    if array.ndim == 2:
        array = np.broadcast_to(array, (p,) + array.shape)
    if (array.ndim != 3 or array.shape[:2] != (p, dimension)
            or not 1 <= array.shape[2] <= dimension):
        raise ValueError("frame must have shape (D,d) or (queries,D,d)")
    gram = np.swapaxes(array, 1, 2) @ array
    if not np.allclose(gram, np.eye(array.shape[2]), atol=1e-10, rtol=1e-9):
        raise ValueError("frame columns must be orthonormal")
    return array


def _hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(array.dtype.str.encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _distances(points, query):
    # Direct differences avoid cancellation from ||x||²+||q||²-2x.q.
    with np.errstate(over="ignore", invalid="ignore"):
        delta = points - query
        distances = np.hypot.reduce(delta, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("coordinate magnitudes overflow float64 distances")
    return distances


def patch_moments(X, neighborhood_ids, frame=None):
    """Measure mean-centered moments on explicit unique row-ID lists.

    The caller supplies nonself lists; this function has no focal identity to
    infer. Different IDs with identical coordinates remain distinct particles.
    ``named_q`` is z variance divided by mean x/y variance when the measured
    frame has exactly three columns (or X has three coordinates and frame=None).
    """
    X = _matrix(X, "X")
    patches = [_ids(ids, len(X)) for ids in neighborhood_ids]
    if any(len(set(ids)) != len(ids) for ids in patches):
        raise ValueError("patch row IDs must be unique")
    p, dimension = len(patches), X.shape[1]
    frames = _frames(frame, p, dimension)
    d = dimension if frames is None else frames.shape[2]
    count = np.array([len(ids) for ids in patches], dtype=np.int64)
    mean = np.full((p, dimension), np.nan)
    centered = np.full((p, dimension, dimension), np.nan)
    unbiased = np.full_like(centered, np.nan)
    named = np.full((p, d, d), np.nan)
    eigenvalues = np.full((p, dimension), np.nan)
    status = np.full(p, "EMPTY", dtype="U32")
    unbiased_status = status.copy()
    q_status = np.full(p, "UNLABELED_FRAME", dtype="U32")
    q = np.full(p, np.nan)
    symmetry_residual = np.full(p, np.nan)
    for a, ids in enumerate(patches):
        m = len(ids)
        if not m:
            q_status[a] = "EMPTY"
            continue
        values = X[ids]
        with np.errstate(over="ignore", invalid="ignore"):
            mu = values.mean(axis=0)
            residuals = values - mu
            cov = residuals.T @ residuals / m
        if not np.isfinite(cov).all() or not np.isfinite(mu).all():
            raise ValueError("coordinate magnitudes overflow float64 moments")
        mean[a], centered[a] = mu, cov
        named[a] = cov if frames is None else frames[a].T @ cov @ frames[a]
        eigenvalues[a] = np.linalg.eigvalsh(cov)
        symmetry_residual[a] = np.max(np.abs(cov - cov.T))
        tolerance = 1e-10 + 1e-9 * np.linalg.norm(cov, 2)
        status[a] = "OK" if eigenvalues[a, 0] >= 0 else "TOLERATED_NEGATIVE_ROUNDOFF"
        if eigenvalues[a, 0] < -tolerance or symmetry_residual[a] > tolerance:
            status[a] = "NUMERICAL_ERROR"
        if m >= 2:
            unbiased[a] = cov * (m / (m - 1))
            unbiased_status[a] = status[a]
        else:
            unbiased_status[a] = "INSUFFICIENT_SUPPORT"
        if d == 3:
            denominator = (named[a, 0, 0] + named[a, 1, 1]) / 2
            threshold = 1e-14 * max(np.trace(cov), np.finfo(np.float64).tiny)
            if denominator <= threshold:
                q_status[a] = "DEGENERATE_TANGENT_SPREAD"
            else:
                q[a] = named[a, 2, 2] / denominator
                q_status[a] = "LOW_SUPPORT" if m < 10 else "OK"
    return dict(count=count, mean=mean, centered_moment=centered,
                unbiased_covariance=unbiased, covariance_status=status,
                unbiased_status=unbiased_status, frame_moment=named,
                eigenvalues=eigenvalues, symmetry_residual=symmetry_residual,
                named_q=q, named_q_status=q_status)


def _secants(values, radii, zero_status):
    result = np.full((values.shape[0], len(radii) - 1), np.nan)
    valid = (values[:, :-1] > 0) & (values[:, 1:] > 0)
    statuses = np.where(valid, "OK", zero_status)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_step = np.diff(radii) / radii[:-1]
        log_step = np.log1p(relative_step)
        log_step = np.where(np.isfinite(log_step), log_step, np.diff(np.log(radii)))
        raw = np.diff(np.log(values), axis=1) / log_step
    result[valid] = raw[valid]
    return result, statuses


def neighborhood_profiles(X, queries, query_ids, radii, *,
                          selection_coordinates=None, selection_queries=None,
                          frame=None, query_block_size=64):
    """Closed-ball counts/moments and ambient Gaussian mass at arbitrary queries.

    Distances and row IDs are ragged lists, sorted by (distance,row ID).
    ``inclusion_offsets[a,l]`` is the prefix length in selection_sorted_ids[a].
    The Gaussian uses ambient distances even when moments use a chart window.
    Only an explicit focal row identity is removed. Returned moments have leading
    axes (query,radius); NaNs represent missing values with reason arrays.
    """
    X, queries, radii = _matrix(X, "X"), _matrix(queries, "queries"), _radii(radii)
    if X.shape[1] != queries.shape[1]:
        raise ValueError("X and queries coordinate dimensions differ")
    identities = _ids(query_ids, len(X), allow_none=True)
    if len(identities) != len(queries):
        raise ValueError("query_ids length differs from queries")
    if (not isinstance(query_block_size, numbers.Integral)
            or isinstance(query_block_size, (bool, np.bool_)) or query_block_size < 1):
        raise ValueError("query_block_size must be a positive integer")
    if (selection_coordinates is None) != (selection_queries is None):
        raise ValueError("both selection arrays must be supplied together")
    S = X if selection_coordinates is None else _matrix(selection_coordinates, "selection_coordinates")
    Q = queries if selection_queries is None else _matrix(selection_queries, "selection_queries")
    if len(S) != len(X) or len(Q) != len(queries) or S.shape[1] != Q.shape[1]:
        raise ValueError("selection arrays have incompatible shapes")
    frames = _frames(frame, len(queries), X.shape[1])
    for a, identity in enumerate(identities):
        if identity is not None:
            if not np.array_equal(queries[a], X[identity]):
                raise ValueError("query coordinates differ from named observation")
            if not np.array_equal(Q[a], S[identity]):
                raise ValueError("selection query differs from named observation")
    p, L = len(queries), len(radii)
    ambient_ids, ambient_distances, selection_ids, selection_distances = [], [], [], []
    counts = np.zeros((p, L), dtype=np.int64)
    mass = np.zeros((p, L), dtype=np.float64)
    eligible_n = np.array([len(X) - int(i is not None) for i in identities], dtype=np.int64)
    patches = []
    timings = {stage: {"wall_seconds": 0., "process_seconds": 0.}
               for stage in ("distances_sorting_counts", "mass", "moments")}
    all_ids = np.arange(len(X))
    for start in range(0, p, query_block_size):
        for a in range(start, min(p, start + query_block_size)):
            wall_start, process_start = time.perf_counter(), time.process_time()
            ids = all_ids if identities[a] is None else all_ids[all_ids != identities[a]]
            da, ds = _distances(X[ids], queries[a]), _distances(S[ids], Q[a])
            oa, os = np.lexsort((ids, da)), np.lexsort((ids, ds))
            ambient_ids.append(ids[oa]); ambient_distances.append(da[oa])
            selection_ids.append(ids[os]); selection_distances.append(ds[os])
            counts[a] = np.searchsorted(ds[os], radii, side="right")
            timings["distances_sorting_counts"]["wall_seconds"] += time.perf_counter() - wall_start
            timings["distances_sorting_counts"]["process_seconds"] += time.process_time() - process_start
            wall_start, process_start = time.perf_counter(), time.process_time()
            for l, r in enumerate(radii):
                with np.errstate(over="ignore", under="ignore"):
                    mass[a, l] = np.exp(-np.square(da / r)).sum(dtype=np.float64)
                patches.append(ids[os][:counts[a, l]])
            timings["mass"]["wall_seconds"] += time.perf_counter() - wall_start
            timings["mass"]["process_seconds"] += time.process_time() - process_start
    wall_start, process_start = time.perf_counter(), time.process_time()
    expanded_frame = None if frames is None else np.repeat(frames, L, axis=0)
    moments = patch_moments(X, patches, expanded_frame)
    moments = {key: value.reshape((p, L) + value.shape[1:]) for key, value in moments.items()}
    timings["moments"]["wall_seconds"] = time.perf_counter() - wall_start
    timings["moments"]["process_seconds"] = time.process_time() - process_start
    normalized_mass = np.full_like(mass, np.nan)
    np.divide(mass, eligible_n[:, None], out=normalized_mass, where=eligible_n[:, None] > 0)
    count_secants, count_status = _secants(counts, radii, "ZERO_COUNT")
    mass_secants, mass_status = _secants(mass, radii, "KERNEL_UNDERFLOW")
    empty_population = eligible_n == 0
    mass_status[empty_population] = "EMPTY_POPULATION"
    mass_flags = np.where(mass == 0, "KERNEL_UNDERFLOW", "OK")
    mass_flags[empty_population] = "EMPTY_POPULATION"
    return dict(eligible_n=eligible_n, ambient_sorted_ids=ambient_ids,
                ambient_sorted_distances=ambient_distances,
                selection_sorted_ids=selection_ids, selection_sorted_distances=selection_distances,
                inclusion_offsets=counts.copy(), hard_count=counts,
                gaussian_mass=mass, normalized_mass=normalized_mass,
                hard_count_zero=counts == 0, mass_status=mass_flags,
                count_secants=count_secants, count_secant_status=count_status,
                mass_secants=mass_secants, mass_secant_status=mass_status,
                secant_radii=np.exp((np.log(radii[:-1]) + np.log(radii[1:])) / 2),
                moments=moments, timings=timings, metadata=dict(
                    window="ambient" if selection_coordinates is None else "selection_coordinates",
                    kernel="exp(-(ambient_distance/bandwidth)**2)", closed_ball=True,
                    focal_exclusion="row_identity", sort_order="distance_then_row_id",
                    dtype="float64", X_sha256=_hash(X), queries_sha256=_hash(queries),
                    query_ids=identities, radii_sha256=_hash(radii),
                    selection_sha256=_hash(S), selection_queries_sha256=_hash(Q),
                    frame_sha256=None if frame is None else _hash(np.asarray(frame))))


def transported_patch_radii(Y, query_ids, neighborhood_ids):
    """Maximum distance to patch and minimum distance to its nonself complement.

    Strict inner_radius < outer_radius means some closed ball selects exactly
    the patch. Exact equality is a boundary tie and cannot separate the patch.
    Empty patch or complement has missing radii and an explicit status.
    """
    Y = _matrix(Y, "Y")
    identities = _ids(query_ids, len(Y))
    patches = [_ids(ids, len(Y)) for ids in neighborhood_ids]
    if len(identities) != len(patches):
        raise ValueError("one patch is required per query identity")
    inside, outside = np.full(len(patches), np.nan), np.full(len(patches), np.nan)
    status = np.full(len(patches), "OK", dtype="U32")
    selectable = np.zeros(len(patches), dtype=bool)
    gap = np.full(len(patches), np.nan)
    all_ids = np.arange(len(Y))
    for a, (identity, ids) in enumerate(zip(identities, patches)):
        if len(ids) != len(set(ids)) or identity in ids:
            raise ValueError("patch must contain unique nonself row IDs")
        complement = all_ids[~np.isin(all_ids, ids + [identity])]
        if ids:
            inside[a] = np.max(_distances(Y[ids], Y[identity]))
        if len(complement):
            outside[a] = np.min(_distances(Y[complement], Y[identity]))
        if not ids:
            status[a] = "EMPTY_PATCH"
        elif not len(complement):
            status[a] = "EMPTY_COMPLEMENT"
        else:
            gap[a] = outside[a] - inside[a]
            selectable[a] = gap[a] > 0
            status[a] = "BOUNDARY_TIE" if gap[a] == 0 else ("OK" if gap[a] > 0 else "INTERLEAVED")
    return dict(inner_radius=inside, outer_radius=outside, gap=gap,
                strictly_selectable=selectable, status=status)


def influence_profiles(weights, distances, radii, *, patch_mask=None, self_indices=None):
    """Inspect dense nonnegative influence rows against corresponding distances.

    No symmetry or PSD assumption is made. Self mass is retained, not removed.
    A zero row has undefined distribution, CDF, participation and leakage.
    Row maxima stabilize normalization; overflowing total mass is rejected.
    """
    weights, distances, radii = _matrix(weights, "weights"), _matrix(distances, "distances"), _radii(radii)
    if weights.shape != distances.shape or np.any(weights < 0) or np.any(distances < 0):
        raise ValueError("weights and nonnegative distances must have the same shape")
    p, n = weights.shape
    maximum = weights.max(axis=1)
    valid = maximum > 0
    scaled = np.zeros_like(weights)
    np.divide(weights, maximum[:, None], out=scaled, where=valid[:, None])
    scaled_sum = scaled.sum(axis=1)
    with np.errstate(over="ignore"):
        row_sum = maximum * scaled_sum
    if not np.isfinite(row_sum).all():
        raise ValueError("row sums overflow float64")
    distribution = np.full_like(weights, np.nan)
    np.divide(scaled, scaled_sum[:, None], out=distribution, where=valid[:, None])
    cdf = np.full((p, len(radii)), np.nan)
    for l, r in enumerate(radii):
        cdf[valid, l] = np.sum(distribution[valid] * (distances[valid] <= r), axis=1)
    participation = np.full(p, np.nan)
    participation[valid] = 1 / np.square(distribution[valid]).sum(axis=1)
    result = dict(row_sum=row_sum, distribution=distribution, radial_cdf=cdf,
                  participation=participation, status=np.where(valid, "OK", "ZERO_ROW"))
    if self_indices is not None:
        ids = _ids(self_indices, n, allow_none=True)
        if len(ids) != p:
            raise ValueError("one self index is required per influence row")
        raw, normalized = np.full(p, np.nan), np.full(p, np.nan)
        for a, identity in enumerate(ids):
            if identity is not None:
                raw[a] = weights[a, identity]
                normalized[a] = distribution[a, identity]
        result.update(self_mass=raw, self_fraction=normalized,
                      self_status=np.array(["EXTERNAL_QUERY" if i is None else ("OK" if valid[a] else "ZERO_ROW")
                                            for a, i in enumerate(ids)]))
    if patch_mask is not None:
        mask = np.asarray(patch_mask)
        if mask.dtype != np.bool_ or mask.shape != weights.shape:
            raise ValueError("patch_mask must be a boolean matrix matching weights")
        leakage = np.full(p, np.nan)
        leakage[valid] = np.sum(distribution[valid] * ~mask[valid], axis=1)
        result["patch_leakage"] = leakage
    return result
