"""Exact finite coordinate selections; no latent-manifold or optimal-layout claim.

Closed quadratic regions are anchored at a particle but exclude that particle by
identity from counts. Coincident other particles remain. Float64 computed values
are compared exactly, without rounding, epsilon expansion, or near-tie grouping.
"""
from __future__ import annotations

import numpy as np
from manylatents.metrics.registry import register_metric


def _real(value, name, ndim=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f'{name} must be real numeric')
    result = np.asarray(raw, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError(f'{name} has invalid shape or nonfinite values')
    return result


def _cloud(value):
    result = _real(value, 'coordinates', 2)
    if min(result.shape) < 1:
        raise ValueError('coordinates must have nonempty rows and dimensions')
    return result


def _probe(index, n):
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < n:
        raise ValueError('probe_index must be a valid integer particle ID')
    return int(index)


def _mask(value, n, name):
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype != np.bool_:
        raise ValueError(f'{name} must be a boolean vector of length n')
    return value.copy()


def _scalar(value, name, nonnegative=False):
    a = _real(value, name, 0)
    x = float(a)
    if nonnegative and x < 0:
        raise ValueError(f'{name} must be nonnegative')
    return x


def _finite(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f'{name} produced a nonfinite value')
    return value


def _collision(Y, target, counted):
    ids = np.flatnonzero(counted)
    _, labels = np.unique(Y[ids], axis=0, return_inverse=True)
    classes = int(labels.max()) + 1 if labels.size else 0
    positive = np.bincount(labels[target[ids]], minlength=classes).astype(np.int64)
    negative = np.bincount(labels[~target[ids]], minlength=classes).astype(np.int64)
    class_id = np.full(len(Y), -1, dtype=np.int64)
    class_id[ids] = labels
    floor = np.minimum(positive, negative)
    return dict(class_id=class_id, positive_counts=positive, negative_counts=negative,
                class_floor=floor, floor=int(floor.sum()), equality='EXACT_COORDINATE_EQUALITY')


def _frontier(tp, fp, indices):
    """Count Pareto candidates in O(C log C), preserving input order and ties.

    Within an equal-FP group only maximal TP can survive. That maximum survives
    exactly when it exceeds every maximum at smaller FP. All occurrences of
    tied candidate IDs are retained, including repeated indices in the input.
    Counts and candidate memberships are never modified by this search.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if not indices.size:
        return indices.copy()
    order = np.argsort(fp[indices], kind='stable')
    ordered_fp = fp[indices[order]]
    ordered_tp = tp[indices[order]]
    starts = np.r_[0, np.flatnonzero(ordered_fp[1:] != ordered_fp[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(indices)])
    maxima = np.maximum.reduceat(ordered_tp, starts)
    surviving_group = np.r_[True, maxima[1:] > np.maximum.accumulate(maxima)[:-1]]
    keep_sorted = ((ordered_tp == np.repeat(maxima, sizes)) &
                   np.repeat(surviving_group, sizes))
    keep_original = np.empty(len(indices), dtype=bool)
    keep_original[order] = keep_sorted
    return indices[keep_original]


def _summary(curve, target_count, full):
    candidates, statuses = [], []
    for start, stop in zip(curve['offsets'][:-1], curve['offsets'][1:]):
        ids = np.arange(start, stop, dtype=np.int64)
        if target_count == 0:
            candidates.append(None); statuses.append('TARGET_EMPTY'); continue
        if full:
            ids = ids[curve['fn'][ids] == 0]
        elif ids.size:
            ids = ids[curve['tp'][ids] == curve['tp'][ids].max()]
        if not ids.size:
            candidates.append(None); statuses.append('UNATTAINABLE' if full else 'NO_CANDIDATES'); continue
        order = np.lexsort((ids, curve['threshold'][ids], curve['fp'][ids]))
        candidates.append(int(ids[order[0]])); statuses.append('OK')
    valid = np.asarray([i is not None for i in candidates], dtype=bool)
    result = dict(candidate_index=candidates, defined=valid, status=statuses)
    for key in ('tp', 'fp', 'fn', 'tn', 'recovery', 'contamination', 'recovery_defined', 'contamination_defined', 'threshold'):
        dtype = curve[key].dtype
        result[key] = np.asarray([curve[key][i] if i is not None else 0 for i in candidates], dtype=dtype)
    return result


def _curve(q, slab_eligible, slab_boundary, target, counted, threshold_lists, exhaustive):
    nforms, n = q.shape
    form_index = np.concatenate([np.full(len(t), f, dtype=np.int64) for f, t in enumerate(threshold_lists)])
    threshold = np.concatenate(threshold_lists)
    offsets = np.concatenate(([0], np.cumsum([len(t) for t in threshold_lists]))).astype(np.int64)
    selected = (q[form_index] <= threshold[:, None]) & slab_eligible[form_index] & counted
    target_count = int(target.sum())
    tp = (selected & target).sum(axis=1, dtype=np.int64)
    fp = (selected & ~target).sum(axis=1, dtype=np.int64)
    fn = target_count - tp
    tn = int(counted.sum()) - target_count - fp
    total = tp + fp
    recovery_defined = np.full(len(tp), target_count > 0, dtype=bool)
    contamination_defined = total > 0
    recovery = np.divide(tp, target_count, out=np.zeros(len(tp), dtype=np.float64), where=recovery_defined)
    contamination = np.divide(fp, total, out=np.zeros(len(tp), dtype=np.float64), where=contamination_defined)
    tied = (q[form_index] == threshold[:, None]) & counted
    # Boundary of the intersected closed region: quadratic face inside slab,
    # or either slab face inside the quadratic region. Count each particle once.
    boundary = selected & (tied | slab_boundary[form_index])
    result = dict(form_index=form_index, threshold=threshold, offsets=offsets,
                  selected_mask=selected, tp=tp, fp=fp, fn=fn, tn=tn,
                  recovery=recovery, contamination=contamination,
                  recovery_defined=recovery_defined, contamination_defined=contamination_defined,
                  tie_count=tied.sum(axis=1, dtype=np.int64),
                  boundary_count=boundary.sum(axis=1, dtype=np.int64),
                  target_empty=~recovery_defined, selection_empty=~contamination_defined,
                  selection_full=total == int(counted.sum()), exhaustive=bool(exhaustive))
    result['best_full_recovery'] = _summary(result, target_count, True)
    result['max_recovery'] = _summary(result, target_count, False)
    result['frontier_space'] = 'TP_MAX_FP_MIN'
    result['count_frontier_indices'] = _frontier(tp, fp, np.arange(len(tp)))
    result['per_form_count_frontier_indices'] = [_frontier(tp, fp, np.arange(offsets[f], offsets[f+1])) for f in range(nforms)]
    return result


@register_metric(aliases=['selection_region_profile'])
def SelectionRegionProfile(embeddings, target_mask, probe_index, forms, slabs=None,
                           thresholds=None, allow_semidefinite=None, *,
                           dataset=None, module=None, cache=None):
    """Enumerate every closed anchored selection in a finite quadratic family.

    ``forms`` is (F,d,d). Each must be exactly symmetric and positive definite;
    ``allow_semidefinite`` is an explicit length-F boolean vector admitting PSD
    forms (e.g. infinite cylinders), without eigenvalue clipping. ``slabs`` is
    None or length F, each None or {normal, offset, halfwidth}; the predicate is
    abs(normal @ y - offset) <= halfwidth, with no normalization of supplied units.

    Primary thresholds are sorted unique {0} union q over ALL counted particles,
    regardless of slab eligibility. Additional ``thresholds`` is length F of
    finite nonnegative vectors; sorted unique evaluations live in ``additional``
    and never replace primary curves. No artificial empty selector is inserted.

    Ratio zeros with false *_defined masks are placeholders, never observations.
    Boundary count refers to the intersected region; tie_count counts q equality
    over all counted particles, including those outside the slab. All frontier
    indices retain tied candidates and pool precisely the supplied forms.
    These count frontiers maximize TP and minimize FP (TP_MAX_FP_MIN); they are
    not recovery/contamination frontiers. Complete attainable curves are retained.
    """
    Y = _cloud(embeddings)
    n, d = Y.shape
    probe = _probe(probe_index, n)
    target = _mask(target_mask, n, 'target_mask')
    counted = np.ones(n, dtype=bool); counted[probe] = False; target[probe] = False
    matrices = _real(forms, 'forms', 3)
    if matrices.shape[0] < 1 or matrices.shape[1:] != (d, d):
        raise ValueError('forms must have shape (F,d,d), F>=1')
    F = len(matrices)
    semidefinite = np.zeros(F, dtype=bool) if allow_semidefinite is None else _mask(allow_semidefinite, F, 'allow_semidefinite')
    eigenvalues = np.empty((F,d))
    for f, M in enumerate(matrices):
        if not np.array_equal(M, M.T):
            raise ValueError('forms must be exactly symmetric; no implicit symmetrization')
        try:
            eigenvalues[f] = np.linalg.eigvalsh(M)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError('form eigensolver failed') from exc
        _finite(eigenvalues[f], 'form eigenvalues')
        if (eigenvalues[f] < 0).any() or (not semidefinite[f] and (eigenvalues[f] <= 0).any()):
            raise ValueError('form must be SPD, or PSD with explicit opt-in')
    if slabs is None:
        slabs = [None] * F
    if not isinstance(slabs, (list, tuple, np.ndarray)) or (isinstance(slabs, np.ndarray) and slabs.ndim == 0) or len(slabs) != F:
        raise ValueError('slabs must have one entry per form')
    eligible = np.ones((F,n), dtype=bool)
    slab_boundary = np.zeros((F,n), dtype=bool)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        centered = _finite(Y-Y[probe], 'centered coordinates')
        q = _finite(np.einsum('ni,fij,nj->fn', centered, matrices, centered), 'quadratic distances')
        if (q < 0).any():
            raise FloatingPointError('negative computed quadratic distance; no clipping is applied')
        for f, slab in enumerate(slabs):
            if slab is None:
                continue
            if not isinstance(slab, dict) or set(slab) != {'normal','offset','halfwidth'}:
                raise ValueError('slab needs exactly normal, offset, halfwidth')
            normal = _real(slab['normal'], 'slab normal', 1)
            if normal.shape != (d,) or not np.any(normal != 0):
                raise ValueError('slab normal must be nonzero with dimension d')
            offset = _scalar(slab['offset'], 'slab offset')
            width = _scalar(slab['halfwidth'], 'slab halfwidth', True)
            distance = _finite(np.abs(Y @ normal - offset), 'slab displacement')
            eligible[f] = distance <= width
            slab_boundary[f] = distance == width
    primary_thresholds = [np.unique(np.concatenate(([0.], row[counted]))) for row in q]
    result = _curve(q, eligible, slab_boundary, target, counted, primary_thresholds, True)
    ball_forms = [f for f,M in enumerate(matrices) if slabs[f] is None and np.array_equal(M,np.eye(d))]
    ball_ids = np.concatenate([np.arange(result['offsets'][f],result['offsets'][f+1]) for f in ball_forms]) if ball_forms else np.empty(0,dtype=np.int64)
    result['ball_count_frontier_indices'] = _frontier(result['tp'],result['fp'],ball_ids)
    result.update(q=q, counted_mask=counted, target_mask=target, slab_eligible=eligible,
                  anchor=Y[probe].copy(), forms=matrices.copy(), eigenvalues=eigenvalues,
                  collision=_collision(Y,target,counted), additional=None,
                  target_count=int(target.sum()), counted_count=int(counted.sum()),
                  self_policy='EXCLUDE_FOCAL_ID', boundary_policy='CLOSED_EXACT',
                  threshold_policy='ZERO_UNION_ALL_COUNTED_QUADRATIC_VALUES')
    if thresholds is not None:
        if not isinstance(thresholds, (list, tuple, np.ndarray)) or (isinstance(thresholds, np.ndarray) and thresholds.ndim == 0) or len(thresholds) != F:
            raise ValueError('thresholds must have one vector per form')
        extra = []
        for values in thresholds:
            t = _real(values, 'additional thresholds', 1)
            if (t < 0).any():
                raise ValueError('additional thresholds must be nonnegative')
            extra.append(np.unique(t))
        result['additional'] = _curve(q,eligible,slab_boundary,target,counted,extra,False)
    return result


@register_metric(aliases=['ball_membership'])
def BallMembership(coordinates=None, probe_index=None, radius=None, eligible_mask=None,
                   *, embeddings=None, dataset=None, module=None, cache=None):
    """Closed source ball, optionally intersected with declared eligibility.

    physical_mask includes eligible focal identity; target_mask clears only that
    identity and preserves coordinate duplicates. Radius zero is supported.
    ``embeddings`` is only the registry-compatible alias for ``coordinates``.
    """
    if coordinates is not None and embeddings is not None:
        raise ValueError('supply coordinates or embeddings, not both')
    X = _cloud(embeddings if coordinates is None else coordinates)
    probe = _probe(probe_index,len(X)); r = _scalar(radius,'radius',True)
    eligible = np.ones(len(X),dtype=bool) if eligible_mask is None else _mask(eligible_mask,len(X),'eligible_mask')
    with np.errstate(over='raise',invalid='raise'):
        distances = _finite(np.hypot.reduce(X-X[probe],axis=1),'ball distances')
    physical = (distances <= r) & eligible
    target = physical.copy(); target[probe] = False
    return dict(distances=distances,physical_mask=physical,target_mask=target,
                eligible_mask=eligible, radius=r,probe_index=probe,
                count=int(target.sum()),physical_count=int(physical.sum()),
                boundary_count=int(((distances==r)&target).sum()),
                self_policy='EXCLUDE_FOCAL_ID',boundary_policy='CLOSED_EXACT')


@register_metric(aliases=['declared_edge_profile'])
def DeclaredEdgeProfile(embeddings, edges, display_radius, *, dataset=None,module=None,cache=None):
    """Lengths and retention of declared unique undirected, non-self ID pairs."""
    Y = _cloud(embeddings)
    pairs = np.asarray(edges)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype,np.integer) or np.issubdtype(pairs.dtype,np.bool_):
        raise ValueError('edges must be an integer (E,2) array')
    if (pairs < 0).any() or (pairs >= len(Y)).any() or (pairs[:,0]==pairs[:,1]).any():
        raise ValueError('edge IDs must be valid and distinct')
    canonical = np.sort(pairs,axis=1)
    if len(np.unique(canonical,axis=0)) != len(pairs):
        raise ValueError('undirected edges must be unique')
    radius = _scalar(display_radius,'display_radius',True)
    with np.errstate(over='raise',invalid='raise'):
        distances = _finite(np.hypot.reduce(Y[pairs[:,0]]-Y[pairs[:,1]],axis=1),'edge distances')
    retained = distances <= radius
    return dict(edges=pairs.astype(np.int64,copy=True),distances=distances,
                retained_mask=retained,within_radius=retained.copy(),lost_mask=~retained,
                retained_count=int(retained.sum()),lost_count=int((~retained).sum()),
                boundary_count=int((distances==radius).sum()),display_radius=radius,
                boundary_policy='CLOSED_EXACT')


def SelectionGaugeProfile(reference_q, transformed_q, *, counted_mask, atol=1e-10, rtol=1e-9):
    """Registered-protocol gauge readout, with explicit safe-threshold coverage.

    Compare all continuous q entries. Midpoints use only counted reference values
    and gaps strictly greater than 2*max_abs_error_for_form + atol. Additional
    membership comparisons must be recomputed by SelectionRegionProfile; this
    helper never supplies or reuses memberships.
    """
    reference = _real(reference_q,'reference_q',2)
    transformed = _real(transformed_q,'transformed_q',2)
    if reference.shape != transformed.shape or min(reference.shape) < 1:
        raise ValueError('gauge arrays must have matching nonempty (F,n) shapes')
    if (reference < 0).any() or (transformed < 0).any():
        raise ValueError('gauge quadratic values must be nonnegative')
    counted = _mask(counted_mask,reference.shape[1],'counted_mask')
    atol = _scalar(atol,'atol',True); rtol = _scalar(rtol,'rtol',True)
    with np.errstate(over='raise',invalid='raise'):
        error = _finite(np.abs(reference-transformed),'gauge error')
        maximum = error.max(axis=1)
        agreement = (error <= _finite(atol + rtol*np.abs(reference),'gauge tolerance')).all(axis=1)
        thresholds = []
        for f,row in enumerate(reference):
            values = np.unique(row[counted])
            gaps = np.diff(values)
            safe = gaps > 2*maximum[f]+atol
            midpoint = values[:-1] + gaps/2
            # An admissible midpoint must also be representably inside its gap.
            safe &= (midpoint > values[:-1]) & (midpoint < values[1:])
            thresholds.append(midpoint[safe])
    coverage = np.asarray([len(t) for t in thresholds],dtype=np.int64)
    return dict(max_abs_error=maximum,q_agreement=agreement,
                gap_thresholds=thresholds,per_form_coverage=coverage,
                eligible_count=int(coverage.sum()),atol=atol,rtol=rtol,
                counted_mask=counted)
