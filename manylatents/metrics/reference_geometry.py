"""Per-point scores of query points against a separate reference cloud.

These are the comparison scores for reference-cloud LID: how far a point is
from the reference (kNN distance, LOF), how much of it lies outside the
reference's leading linear subspace (PCA), and how many directions its
reference neighbourhood spans (participation ratio). None of them uses labels,
and the query points never serve as each other's neighbours.

They are library functions, not registered metrics: each needs a second array
that the metric protocol has no slot for.
"""
from __future__ import annotations

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable, unavailable_for_points
from manylatents.utils.knn import compute_knn, compute_knn_query


def _pair(query, reference) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for name, x in (("query", query), ("reference", reference)):
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
        arrays.append(np.asarray(x, dtype=np.float64))
    q, r = arrays
    if q.shape[1] != r.shape[1]:
        raise MeasurementUnavailable(
            f"reference has {r.shape[1]} features, query has {q.shape[1]}"
        )
    return q, r


def knn_distance_score(query: np.ndarray, reference: np.ndarray, k: int = 20) -> np.ndarray:
    """Mean Euclidean distance from each query row to its k nearest reference rows."""
    q, r = _pair(query, reference)
    distances, _ = compute_knn_query(r, q, k)
    return distances.mean(axis=1)


def lof_novelty_score(query: np.ndarray, reference: np.ndarray, k: int = 20) -> np.ndarray:
    """Local outlier factor of each query row relative to the reference cloud.

    About 1 for a point as dense as its reference neighbours, larger for a
    point in a sparser spot. The reference alone defines the densities.
    """
    q, r = _pair(query, reference)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 < k < r.shape[0]:
        raise MeasurementUnavailable(
            f"LOF requires 0 < k < n_reference; got k={k}, n_reference={r.shape[0]}"
        )
    # Reference densities use other reference rows, as in novelty-mode LOF.
    ref_distances, ref_indices = compute_knn(r, k=int(k), include_self=False)
    k_distance = ref_distances[:, -1]
    ref_reach = np.maximum(ref_distances, k_distance[ref_indices]).mean(axis=1)
    distances, indices = compute_knn_query(r, q, int(k))
    query_reach = np.maximum(distances, k_distance[indices]).mean(axis=1)
    neighbour_reach = ref_reach[indices]
    bad = (~np.isfinite(query_reach) | (query_reach <= 0)
           | ~np.all(np.isfinite(neighbour_reach) & (neighbour_reach > 0), axis=1))
    if bad.any():
        raise unavailable_for_points("LOF reachability is not finite and positive", bad)
    # lrd(neighbour) / lrd(query) = query_reach / neighbour_reach.
    scores = (query_reach[:, None] / neighbour_reach).mean(axis=1)
    bad = ~np.isfinite(scores)
    if bad.any():
        raise unavailable_for_points("LOF is not finite", bad)
    return scores


def pca_reference_scores(
    query: np.ndarray, reference: np.ndarray, n_components: int
) -> dict[str, np.ndarray]:
    """Split each centred query row into its part inside and outside the
    reference's leading principal subspace.

    Returns ``{"leading_norm": ..., "residual_norm": ...}``, each (n_query,).
    The subspace comes from a thin SVD of the centred reference, so this works
    when there are more features than reference points.
    """
    q, r = _pair(query, reference)
    # Translate before averaging so identical decimal rows remain exactly zero.
    shifted = r - r[0]
    mean_shift = shifted.mean(axis=0)
    centred_reference = shifted - mean_shift
    if not np.isfinite(centred_reference).all():
        raise MeasurementUnavailable("centred reference is not finite")
    _, singular, vt = np.linalg.svd(centred_reference, full_matrices=False)
    tolerance = singular.max(initial=0.0) * max(r.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > tolerance))
    if (isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer))
            or not 0 < n_components <= rank):
        raise MeasurementUnavailable(
            f"n_components must be an integer in 1..{rank} (the reference rank); "
            f"got {n_components}"
        )
    basis = vt[: int(n_components)]
    centred = (q - r[0]) - mean_shift
    coordinates = centred @ basis.T
    residual = centred - coordinates @ basis
    scores = {
        "leading_norm": np.linalg.norm(coordinates, axis=1),
        "residual_norm": np.linalg.norm(residual, axis=1),
    }
    bad = ~(np.isfinite(scores["leading_norm"]) & np.isfinite(scores["residual_norm"]))
    if bad.any():
        raise unavailable_for_points("PCA norms are not finite", bad)
    return scores


def local_participation_ratio(
    query: np.ndarray, reference: np.ndarray, k: int = 20, chunk_size: int = 256
) -> np.ndarray:
    """Participation ratio of the spectrum of each query's k reference neighbours.

    ``(sum lambda)**2 / sum(lambda**2)`` over the eigenvalues of the centred
    neighbourhood: about the number of directions the neighbourhood spans.
    The query row itself is not part of the neighbourhood. Eigenvalues come
    from the k-by-k Gram matrix, so the cost does not grow with the feature
    count beyond one matrix product.
    """
    q, r = _pair(query, reference)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2:
        raise MeasurementUnavailable("k must be a nonboolean integer >= 2")
    if (isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise MeasurementUnavailable("chunk_size must be a positive nonboolean integer")
    _, indices = compute_knn_query(r, q, int(k))
    out = np.empty(q.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], int(chunk_size)):
        stop = start + int(chunk_size)
        neighbours = r[indices[start:stop]]                       # (c, k, d)
        neighbours = neighbours - neighbours[:, :1, :]
        neighbours = neighbours - neighbours.mean(axis=1, keepdims=True)
        gram = np.einsum("cid,cjd->cij", neighbours, neighbours)  # (c, k, k)
        eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
        s1 = eigenvalues.sum(axis=1)
        s2 = (eigenvalues ** 2).sum(axis=1)
        out[start:stop] = np.divide(s1 ** 2, s2, out=np.full_like(s1, np.nan), where=s2 > 0)
    bad = ~np.isfinite(out)
    if bad.any():
        raise unavailable_for_points(
            "participation ratio is undefined (neighbourhood has no spread)", bad
        )
    return out


def standardize_against(
    query: np.ndarray, reference: np.ndarray, constant: str = "raise"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score both arrays with the reference's column means and standard deviations.

    Args:
        constant: what to do with columns the reference does not vary in.
            "raise" (default) refuses; "drop" removes them and reports which
            columns were kept.

    Returns:
        (query_z, reference_z, kept): ``kept`` is a boolean mask over the
        original columns.
    """
    if constant not in ("raise", "drop"):
        raise ValueError(f"constant must be 'raise' or 'drop', got {constant!r}")
    q, r = _pair(query, reference)
    mean = r.mean(axis=0)
    std = r.std(axis=0)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise MeasurementUnavailable("reference column moments are not finite")
    kept = np.any(r != r[:1], axis=0)
    if np.any(kept & (std == 0)):
        raise MeasurementUnavailable("nonconstant reference columns have unresolved variance")
    if not kept.all():
        if constant == "raise":
            raise MeasurementUnavailable(
                f"{int((~kept).sum())} of {kept.size} columns are constant in the "
                "reference and cannot be standardized"
            )
        if not kept.any():
            raise MeasurementUnavailable("every column is constant in the reference")
    query_z = (q[:, kept] - mean[kept]) / std[kept]
    reference_z = (r[:, kept] - mean[kept]) / std[kept]
    for name, values in (("query", query_z), ("reference", reference_z)):
        bad = ~np.isfinite(values).all(axis=1)
        if bad.any():
            raise unavailable_for_points(f"standardized {name} is not finite", bad)
    return query_z, reference_z, kept
