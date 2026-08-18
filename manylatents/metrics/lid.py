import logging
from typing import Optional, Tuple, Union

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


def _rms_normalise(embeddings: np.ndarray) -> np.ndarray:
    """Scale an embedding matrix to unit elementwise RMS, in float64.

    LID is invariant to a positive rescale in exact arithmetic, so this is a
    no-op mathematically. It is not a no-op numerically: ``compute_knn`` works
    in float32 and FAISS computes ``||a||^2 + ||b||^2 - 2a.b``, which cancels
    away entirely once activations reach ~1e-16. Dividing by one finite nonzero
    constant moves any such layer into float32's usable range without touching
    the geometry.
    """
    if hasattr(embeddings, "detach"):  # torch tensor, possibly on GPU
        embeddings = embeddings.detach().cpu().numpy()
    x = np.asarray(embeddings, dtype=np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    if not np.isfinite(rms) or rms == 0.0:
        raise ValueError(
            f"cannot normalise embeddings with RMS {rms!r}: the matrix is "
            "all-zero or non-finite, so it carries no geometry to measure"
        )
    return x / rms


def _distinct_rows(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group bit-identical rows. Returns ``(unique_rows, inverse)``.

    Compares raw bytes through a void view, so two rows are "the same point"
    only if they are bit-identical — no tolerance, nothing to tune. Run on the
    float32 array that ``compute_knn`` will actually see, so rows that only
    coincide after the float32 cast are caught here rather than downstream.
    """
    xc = np.ascontiguousarray(x).copy()
    xc[xc == 0] = 0.0  # collapse -0.0, which differs in bytes but not in value
    void = xc.view(np.dtype((np.void, xc.dtype.itemsize * xc.shape[1]))).ravel()
    _, index, inverse = np.unique(void, return_index=True, return_inverse=True)
    return xc[index], np.asarray(inverse).ravel()


@register_metric(
    aliases=["local_intrinsic_dim", "lid", "intrinsic_dim"],
    default_params={"return_per_sample": False},
    description="Mean local intrinsic dimensionality of the embedding",
)
def LocalIntrinsicDimensionality(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: int = 20,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
    dedup: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute the Local Intrinsic Dimensionality (LID) for the embedding.

    Levina-Bickel maximum likelihood estimator over k-NN distances:
    ``-k / sum_j log(d_j / r_k)``. kNN is computed via FAISS when available
    (~10-50x faster), sklearn otherwise.

    The embedding is RMS-normalised and deduplicated before any distance work.
    Both steps are exact no-ops on a well-conditioned, duplicate-free cloud;
    without them the estimator silently reports the value of an epsilon rather
    than a dimension. See the note below.

    Parameters:
      - embeddings: A numpy array representing the embeddings.
      - dataset: Provided for protocol compliance (unused).
      - module: Provided for protocol compliance (unused).
      - k: The number of nearest neighbors to consider.
      - return_per_sample: If True, return per-sample LID values; else return mean.
      - cache: Optional shared cache dict. Passed through to compute_knn().
        Note that the cache is keyed by content, so LID's normalised copy gets
        its own entry rather than sharing one with metrics that consume the raw
        embedding.
      - dedup: Build the neighbour bank from bit-distinct rows and give every
        row the LID of its own point. Present so the no-op property is
        testable, not as a tuning knob: with duplicates in the cloud and dedup
        disabled the call raises rather than returning a number, because that
        is the case the removed epsilons used to paper over.

    Returns:
      - float: Mean LID (if return_per_sample=False)
      - np.ndarray: Per-sample LID values (if return_per_sample=True)

    Raises:
      - ValueError: if the embedding has no scale (all-zero or non-finite RMS),
        if fewer than k+1 distinct points remain, or if a neighbour distance is
        zero between rows that are not bit-identical. Each of these used to
        return a plausible number instead.

    Note:
        This estimator previously clamped ``r_k`` to ``1e-10`` and added
        ``1e-10`` inside the log "to prevent division by zero (duplicate
        embeddings)". Both are absolute constants applied to a scale-free
        quantity, and they decided the answer rather than protecting it:

        * an embedding at RMS 7.6e-16 has every distance below the clamp, so
          the log-ratios collapse and LID falls below 1;
        * a row with j exact duplicates among its k neighbours contributes j
          terms of ``log(1e-10 / r_k)`` — a value fixed by the constant, not by
          the geometry — and is *not* dropped. At 60% duplicates the reported
          median LID moves 0.766 -> 0.599 -> 0.492 for an epsilon of
          1e-9 / 1e-12 / 1e-15.

        Deduplicating removes the zero distances the epsilons existed to guard,
        so the epsilons are gone. Duplicate rows are the same point and receive
        the same LID; when such a group spans classes, that shared value is the
        honest answer and any ceiling it places on a downstream statistic is a
        property of the embedding.

        A median LID below 1 in a high-dimensional space is not a dimension —
        it remains a useful canary that an estimator is out of its domain.
    """
    x = _rms_normalise(embeddings).astype(np.float32)

    if dedup:
        points, inverse = _distinct_rows(x)
    else:
        points, inverse = x, np.arange(x.shape[0])

    n_points = points.shape[0]
    if n_points <= k:
        raise ValueError(
            f"only {n_points} distinct points for k={k}: the cloud has too "
            "little structure to carry a local dimension"
        )

    distances, _ = compute_knn(points, k=k, include_self=False, cache=cache)

    if not np.all(distances > 0):
        raise ValueError(
            "a neighbour distance is zero between rows that are not "
            "bit-identical; the embedding is degenerate at float32 precision "
            "and LID is undefined there"
        )

    r_k = distances[:, -1]
    lid_values = -k / np.sum(np.log(distances / r_k[:, None]), axis=1)
    lid_values = lid_values[inverse]

    if return_per_sample:
        logger.info(
            f"LocalIntrinsicDimensionality: per-sample LID, "
            f"mean={np.mean(lid_values):.3f}, "
            f"{n_points}/{x.shape[0]} distinct points"
        )
        return lid_values

    mean_lid = float(np.mean(lid_values))
    logger.info(
        f"LocalIntrinsicDimensionality: Computed mean LID = {mean_lid:.3f} "
        f"({n_points}/{x.shape[0]} distinct points)"
    )
    return mean_lid
