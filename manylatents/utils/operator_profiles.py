"""Exact-particle influence quantiles and finite-patch propagation diagnostics.

These operations measure supplied objects. They never fit a model or replace an
existing dimensionality, recall, floor, or strain estimator.
"""
from __future__ import annotations

import numbers
import numpy as np

from .neighborhood_profiles import transported_patch_radii


def _matrix(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real-valued; complex coordinates or weights are unsupported')
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite numeric matrix')
    return value


def _identities(query_ids, neighborhood_ids, n):
    def identity(i):
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral) or not 0 <= i < n:
            raise ValueError('Particle identities must be integers in range')
        return int(i)
    queries = [identity(i) for i in query_ids]
    patches = [[identity(i) for i in ids] for ids in neighborhood_ids]
    if len(queries) != len(patches):
        raise ValueError('One nonself patch is required for every query')
    for i, ids in zip(queries, patches):
        if i in ids or len(ids) != len(set(ids)):
            raise ValueError('Patch lists must contain unique nonself particle IDs')
    return queries, patches


def weighted_radius_quantiles(weights, distances, quantiles=(.5, .9)):
    """Return weighted radius quantiles on actual particle distances, including 0.

    Nonnegative weights are normalized independently per row. Ties are sorted
    by distance then particle ID, and the first crossing returns its distance,
    so a closed ball retains all coincident particles. Zero rows return NaN
    with ZERO_ROW. No plotting grid or positive-radius floor is involved.
    """
    weights, distances = _matrix(weights, 'weights'), _matrix(distances, 'distances')
    if weights.shape != distances.shape or weights.shape[1] == 0:
        raise ValueError('Weights and distances must have equal nonempty columns')
    if np.any(weights < 0) or np.any(distances < 0):
        raise ValueError('Weights and distances must be nonnegative')
    qs = np.asarray(quantiles, dtype=np.float64)
    if qs.ndim != 1 or len(qs) == 0 or not np.isfinite(qs).all() or np.any(qs <= 0) or np.any(qs > 1):
        raise ValueError('Quantiles must lie in (0,1]')
    p, n = weights.shape
    radii = np.full((p, len(qs)), np.nan)
    row_sum = np.zeros(p)
    summation_residual = np.zeros(p)
    status = np.full(p, 'ZERO_ROW', dtype='U32')
    ids = np.arange(n)
    for a in range(p):
        maximum = weights[a].max()
        if maximum == 0:
            continue
        scaled = weights[a] / maximum
        order = np.lexsort((ids, distances[a]))
        cumulative = np.cumsum(scaled[order], dtype=np.float64)
        total = cumulative[-1]
        with np.errstate(over='ignore'):
            row_sum[a] = maximum * total
        if not np.isfinite(row_sum[a]):
            raise ValueError('Row mass overflows float64')
        # Ordered cumulative total defines the numerical row normalization;
        # record its discrepancy from a separate reduction, without a pseudocount.
        summation_residual[a] = total - np.sum(scaled, dtype=np.float64)
        crossing = np.searchsorted(cumulative, qs * total, side='left')
        radii[a] = distances[a, order[crossing]]
        status[a] = 'OK'
    return dict(radii=radii, status=status, quantiles=qs, row_sum=row_sum,
                scaled_summation_residual=summation_residual,
                convention='actual distances; closed cumulative crossing; self retained')


def finite_patch_propagation(P, query_ids, neighborhood_ids, *, steps=(1, 5, 15), atol=1e-6, rtol=1e-6):
    """Recompute full-chain and killed-chain diagnostics for nonself patch lists.

    Each physical patch J is the supplied nonself patch union its focal identity.
    Tiny negative entries and row-sum errors within tolerance are corrected to
    the stochastic domain and fully reported. Invalid domains return unavailable
    arrays/statuses; no correction larger than tolerance is applied. Powers of
    raw captured P are retained separately from corrected diagnostic powers.
    """
    P = _matrix(P, 'P')
    if P.shape[0] != P.shape[1] or P.shape[0] == 0:
        raise ValueError('P must be a nonempty square particle matrix')
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    times = list(steps)
    if (not times or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, numbers.Integral) or t < 1 for t in times)
            or len(set(times)) != len(times) or times != sorted(times)):
        raise ValueError('Steps must be distinct positive increasing integers')
    times = np.asarray(times, dtype=np.int64)
    queries, patches = _identities(query_ids, neighborhood_ids, len(P))
    p, T = len(queries), len(times)
    outputs = {key: np.full((p, T), np.nan) for key in
               ('killed_loss', 'endpoint_outside', 'bound_geom', 'bound_lin')}
    checks = np.zeros((p, T), dtype=bool)
    status = np.full(p, 'INVALID_TRANSITION_DOMAIN', dtype='U40')
    delta = np.full(p, np.nan)
    row_sums = P.sum(axis=1)
    row_residual = row_sums - 1
    tolerance = atol + rtol
    raw_powers = {}
    corrections = dict(negative_entries=int(np.sum(P < 0)), max_negative_magnitude=float(max(0., -P.min())),
                       max_row_sum_residual=float(np.max(np.abs(row_residual))),
                       max_entry_correction=0., max_row_l1_correction=0., applied=False,
                       rule='clip tolerated negatives to zero, then normalize tolerated row-sum errors')
    common = dict(steps=times, delta=delta, status=status, checks=checks,
                  raw_powers=raw_powers, row_sum_residual=row_residual,
                  corrections=corrections, query_ids=np.asarray(queries, dtype=np.int64),
                  physical_patch_ids=[np.array(sorted([q] + ids), dtype=np.int64) for q, ids in zip(queries, patches)])
    if np.any(P < -tolerance) or np.any(row_sums <= 0) or np.any(np.abs(row_residual) > tolerance):
        return dict(outputs, **common, powers={}, transition_used=None,
                    monotonicity=np.zeros(p, dtype=bool), bound_status=np.full((p, T), 'UNAVAILABLE', dtype='U32'))
    transition = np.maximum(P, 0)
    transition /= transition.sum(axis=1, keepdims=True)
    correction = np.abs(transition - P)
    corrections['max_entry_correction'] = float(np.max(correction))
    corrections['applied'] = bool(np.any(correction > 0))
    corrections['max_row_l1_correction'] = float(np.max(correction.sum(axis=1)))
    if (np.any(correction > atol + rtol * np.abs(P))
            or np.any(correction.sum(axis=1) > atol + rtol * np.abs(P).sum(axis=1))):
        return dict(outputs, **common, powers={}, transition_used=None,
                    monotonicity=np.zeros(p, dtype=bool), bound_status=np.full((p, T), 'UNAVAILABLE', dtype='U32'))
    raw_powers.update({str(int(t)): np.linalg.matrix_power(P, int(t)) for t in times})
    powers = {str(int(t)): np.linalg.matrix_power(transition, int(t)) for t in times}
    monotonicity = np.ones(p, dtype=bool)
    bound_status = np.full((p, T), 'OK', dtype='U32')
    all_ids = np.arange(len(P))
    for a, (focal, J) in enumerate(zip(queries, common['physical_patch_ids'])):
        outside = all_ids[~np.isin(all_ids, J)]
        Q = transition[np.ix_(J, J)]
        escape = transition[np.ix_(J, outside)].sum(axis=1)
        delta[a] = float(np.max(escape))
        mu = np.zeros(len(J))
        mu[np.flatnonzero(J == focal)[0]] = 1
        saved = {}
        for t in range(1, int(times[-1]) + 1):
            mu = mu @ Q
            if t in times:
                saved[t] = 1 - mu.sum()
        for col, t in enumerate(times):
            killed = float(saved[int(t)])
            endpoint = float(powers[str(int(t))][focal, outside].sum())
            geometric = float(-np.expm1(t * np.log1p(-delta[a]))) if delta[a] < 1 else 1.
            linear = float(t * delta[a])
            outputs['killed_loss'][a, col], outputs['endpoint_outside'][a, col] = killed, endpoint
            outputs['bound_geom'][a, col], outputs['bound_lin'][a, col] = geometric, linear
            chain = np.array([0., endpoint, killed, geometric, linear])
            checks[a, col] = bool(np.isfinite(chain).all() and np.all(chain[:-1] <= chain[1:] + atol + rtol*np.abs(chain[1:])))
            if geometric >= .99:
                bound_status[a, col] = 'NEAR_VACUOUS'
        losses = outputs['killed_loss'][a]
        monotonicity[a] = bool(np.all(np.diff(losses) >= -atol - rtol*np.abs(losses[1:])))
        checks[a] &= monotonicity[a]
        status[a] = 'OK' if np.all(checks[a]) else 'NUMERICAL_CHECK_FAILED'
    return dict(outputs, **common, powers=powers, transition_used=transition,
                monotonicity=monotonicity, bound_status=bound_status)


def transported_patch_summary(Y, query_ids, neighborhood_ids):
    """Add a closed-ball intruder count to the native transported patch radii.

    Intruders are nonself complement particles within the maximum inside radius.
    Empty patches have unavailable counts; full nonself patches have zero.
    """
    Y = _matrix(Y, 'Y')
    queries, patches = _identities(query_ids, neighborhood_ids, len(Y))
    result = transported_patch_radii(Y, queries, patches)
    intruders = np.full(len(queries), np.nan)
    statuses = np.full(len(queries), 'EMPTY_PATCH', dtype='U32')
    all_ids = np.arange(len(Y))
    for a, (q, ids) in enumerate(zip(queries, patches)):
        if not ids:
            continue
        complement = all_ids[~np.isin(all_ids, ids + [q])]
        distances = np.hypot.reduce(Y[complement] - Y[q], axis=1)
        intruders[a] = int(np.sum(distances <= result['inner_radius'][a]))
        statuses[a] = 'OK'
    result.update(intruder_count=intruders, intruder_status=statuses)
    return result
