"""Generalized LID: tail-index estimators on nearest-neighbour distances.

If the number of points within radius ``r`` of a point grows like ``r**m``,
then for ascending neighbour distances ``T_1..T_k`` the log-ratios
``log(T_k / T_j)`` behave like exponential samples with mean ``xi = 1/m``.

* Hill (``method="mle"``): the mean of those log-ratios. ``1/xi`` is exactly
  the Levina-Bickel LID of ``metrics/lid.py``. Always positive.
* Pickands (``method="pickands"``): a sign-bearing estimate from three order
  statistics of the inverse distances. ``xi > 0`` is a power-law neighbourhood,
  ``xi ~ 0`` an exponential one, ``xi < 0`` a bounded one.

Distances come either from within one cloud (transductive, the same
conditioning as ``lid.py``) or from a separate ``reference`` cloud, in which
case the query points never act as each other's neighbours.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from manylatents.metrics.lid import _distinct_rows
from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable, unavailable_for_points
from manylatents.utils.knn import compute_knn, compute_knn_query


def _matrix(x, name: str) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if (
        x.ndim != 2 or 0 in x.shape
        or not np.issubdtype(x.dtype, np.number)
        or np.iscomplexobj(x)
    ):
        raise MeasurementUnavailable(f"{name} must be a nonempty real numeric matrix")
    bad = ~np.isfinite(x).all(axis=1)
    if bad.any():
        raise unavailable_for_points(f"{name} contains non-finite values", bad)
    return np.asarray(x, dtype=np.float64)


def _rms(x: np.ndarray) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rms = float(np.sqrt(np.mean(x ** 2)))
    if not np.isfinite(rms) or rms == 0.0:
        raise MeasurementUnavailable(
            f"cannot normalise a cloud with RMS {rms!r}: no geometry is resolvable"
        )
    return rms


def _row_keys(x32: np.ndarray) -> list[bytes]:
    xc = np.ascontiguousarray(x32).copy()
    xc[xc == 0] = 0  # -0.0 and +0.0 are the same coordinate
    return [row.tobytes() for row in xc]


def _check_k(k) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2:
        raise MeasurementUnavailable("k must be a nonboolean integer >= 2")
    return int(k)


def tail_distances(
    points: np.ndarray,
    k: int = 20,
    reference: Optional[np.ndarray] = None,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """Ascending, strictly positive distances to each point's k nearest neighbours.

    ``reference=None``: neighbours are the other distinct rows of ``points``;
    duplicate observations share their point's distances. This is the
    conditioning ``metrics/lid.py`` uses.

    ``reference`` given: neighbours are distinct rows of ``reference``. A
    reference row identical to the query row is excluded, so a point that also
    sits in the reference cloud is measured against its surroundings, not
    against itself. Both clouds are divided by the reference RMS before the
    float32 search.

    Raises:
        MeasurementUnavailable: fewer than k+1 distinct neighbours available,
            distinct points collapsing at float32, or any non-positive or
            non-finite distance.
    """
    k = _check_k(k)
    x = _matrix(points, "points")

    if reference is None:
        distinct, inverse = _distinct_rows(x)
        n_points = distinct.shape[0]
        conditioned = (distinct / _rms(distinct)).astype(np.float32)
        if _distinct_rows(conditioned)[0].shape[0] != n_points:
            raise MeasurementUnavailable(
                "distinct points collapse at float32 working precision"
            )
        if k >= n_points:
            raise MeasurementUnavailable(
                f"only {n_points} distinct points for k={k}: need at least k+1"
            )
        distances, _ = compute_knn(conditioned, k=k, include_self=False, cache=cache)
        distances = np.asarray(distances, dtype=np.float64)[inverse]
    else:
        ref = _matrix(reference, "reference")
        if ref.shape[1] != x.shape[1]:
            raise MeasurementUnavailable(
                f"reference has {ref.shape[1]} features, points have {x.shape[1]}"
            )
        ref_distinct, _ = _distinct_rows(ref)
        n_reference = ref_distinct.shape[0]
        scale = _rms(ref_distinct)
        ref32 = (ref_distinct / scale).astype(np.float32)
        if _distinct_rows(ref32)[0].shape[0] != n_reference:
            raise MeasurementUnavailable(
                "distinct reference points collapse at float32 working precision"
            )
        if k >= n_reference:
            raise MeasurementUnavailable(
                f"only {n_reference} distinct reference points for k={k}: need at least k+1"
            )
        query32 = (x / scale).astype(np.float32)
        candidates, indices = compute_knn_query(ref32, query32, k + 1)

        # Drop exactly one candidate per query: the coincident reference row if
        # there is one among the candidates, otherwise the farthest.
        row_of = {key: i for i, key in enumerate(_row_keys(ref32))}
        coincident = np.array(
            [row_of.get(key, -1) for key in _row_keys(query32)], dtype=np.int64
        )
        # Float32 equality must not turn a distinct query into an excluded self.
        collapsed = np.array([
            i >= 0 and not np.array_equal(point, ref_distinct[i])
            for point, i in zip(x, coincident)
        ])
        if collapsed.any():
            raise unavailable_for_points("query and reference points collapse at float32 precision", collapsed)
        drop = indices == coincident[:, None]
        drop[~drop.any(axis=1), -1] = True
        distances = candidates[~drop].reshape(x.shape[0], k)

    bad = ~np.all(np.isfinite(distances) & (distances > 0), axis=1)
    if bad.any():
        raise unavailable_for_points(
            "neighbour distances are not finite and positive at float32 precision", bad
        )
    return distances


def _check_distances(distances, min_k: int) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] == 0 or d.shape[1] < min_k:
        raise MeasurementUnavailable(
            f"distances must be (n_points, k) with k >= {min_k}; got shape {d.shape}"
        )
    bad = ~np.all(np.isfinite(d) & (d > 0), axis=1)
    if bad.any():
        raise unavailable_for_points("distances must be finite and strictly positive", bad)
    bad = np.any(np.diff(d, axis=1) < 0, axis=1)
    if bad.any():
        raise unavailable_for_points("distances must be ascending along each row", bad)
    return d


def _log_ratios(d: np.ndarray) -> np.ndarray:
    """log(T_k / T_j) for j < k: k-1 non-negative terms per point."""
    return np.log(d[:, -1:]) - np.log(d[:, :-1])


def hill_tail_index(distances: np.ndarray) -> np.ndarray:
    """Hill estimate of xi per point; ``1 / xi`` is the Levina-Bickel LID."""
    d = _check_distances(distances, min_k=2)
    xi = _log_ratios(d).mean(axis=1)
    bad = ~(np.isfinite(xi) & (xi > 0))
    if bad.any():
        raise unavailable_for_points(
            "Hill tail index is undefined (all neighbour radii equal)", bad
        )
    return xi


def pickands_tail_index(distances: np.ndarray) -> np.ndarray:
    """Pickands estimate of xi per point, from inverse distances. Any sign."""
    d = _check_distances(distances, min_k=4)
    j = d.shape[1] // 4
    t1, t2, t4 = d[:, j - 1], d[:, 2 * j - 1], d[:, 4 * j - 1]
    bad = ~((t2 > t1) & (t4 > t2))
    if bad.any():
        raise unavailable_for_points(
            "Pickands tail index is undefined (tied order statistics)", bad
        )
    # (1/t1 - 1/t2) / (1/t2 - 1/t4), evaluated in log space.
    return (np.log(t2 - t1) - np.log(t4 - t2) + np.log(t4) - np.log(t1)) / np.log(2.0)


def exponentiality(distances: np.ndarray) -> np.ndarray:
    """How exponential the log-ratios look, in [0, 1]; 1 is consistent with a power law.

    Exponential samples have ``mean**2 / mean(square) = 1/2``. The score is
    ``1 - |ratio - 1/2| / (1/4)`` clipped to [0, 1]. A heuristic moment check,
    not a hypothesis test.
    """
    d = _check_distances(distances, min_k=3)
    t = _log_ratios(d)
    m1 = t.mean(axis=1)
    m2 = (t * t).mean(axis=1)
    bad = ~(m2 > 0)
    if bad.any():
        raise unavailable_for_points(
            "exponentiality is undefined (all neighbour radii equal)", bad
        )
    return np.clip(1.0 - np.abs(m1 * m1 / m2 - 0.5) / 0.25, 0.0, 1.0)


@register_metric(
    aliases=["gpd_lid", "generalized_lid", "tail_index"],
    default_params={"return_per_sample": False},
    description="Mean tail index xi of neighbour distances (Hill: xi = 1/LID; Pickands: signed)",
)
def GeneralizedLID(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: Optional[int] = 20,
    method: str = "mle",
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
    reference: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Tail index xi of each point's neighbour distances.

    Args:
        embeddings: (n_samples, n_features) points to score.
        dataset: Provided for protocol compliance (unused).
        module: Provided for protocol compliance (unused).
        k: neighbour count. None resolves to 20. Pickands uses order statistics
            k/4, k/2 and k and is noisy below k of about 100.
        method: "mle" (Hill, xi > 0, 1/xi = LID) or "pickands" (signed).
        return_per_sample: return the (n_samples,) array instead of its mean.
        cache: shared kNN cache, used only without ``reference``.
        reference: optional (n_reference, n_features) cloud. When given,
            neighbours are taken from it and never from ``embeddings``.

    Raises:
        MeasurementUnavailable: see :func:`tail_distances` and the estimators.
            No point is dropped to salvage a mean.
    """
    if k is None:
        k = 20
    if method not in ("mle", "pickands"):
        raise MeasurementUnavailable(
            f"unknown method {method!r}; use 'mle' or 'pickands'"
        )
    distances = tail_distances(embeddings, k=k, reference=reference, cache=cache)
    xi = hill_tail_index(distances) if method == "mle" else pickands_tail_index(distances)
    return xi if return_per_sample else float(np.mean(xi))
