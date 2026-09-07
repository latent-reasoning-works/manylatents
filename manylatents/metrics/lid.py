import logging
from typing import Optional, Tuple, Union

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


def _rms_normalise(embeddings: np.ndarray) -> np.ndarray:
    """Scale to unit elementwise RMS before the float32 kNN computation.

    Positive scaling leaves distance ratios unchanged in exact arithmetic.
    Conditioning keeps tiny activations away from float32 distance underflow.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rms = np.sqrt(np.mean(x ** 2))
    if not np.isfinite(rms) or rms == 0.0:
        raise MeasurementUnavailable(
            f"cannot normalise embeddings with RMS {rms!r}: "
            "no geometry is resolvable at the working precision"
        )
    return x / rms


def _distinct_rows(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group exact rows at the supplied dtype, identifying signed zeros.

    Returns ``(unique_rows, inverse)`` without a tolerance. Call on the original
    data first: newly coincident rows after conditioning are precision loss,
    not duplicate observations.
    """
    xc = np.ascontiguousarray(x).copy()
    xc[xc == 0] = 0  # -0.0 and +0.0 denote the same coordinate
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
    k: Optional[int] = 20,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
    dedup: bool = True,
) -> Union[float, np.ndarray]:
    """Estimate LID using ``-k / sum_j log(d_j / r_k)`` over k-NN distances.

    Neighbors are selected from exact distinct input points, RMS-normalised
    before float32 kNN (FAISS when available, sklearn otherwise). Each duplicate
    observation receives its point's estimate, in original input order. The
    mean therefore weights observations by multiplicity, while neighborhoods
    count distinct points. This is the estimator's duplicate policy.

    Args:
        embeddings: Nonempty real matrix (n_samples, n_features), numpy or torch.
        dataset: Provided for protocol compliance (unused).
        module: Provided for protocol compliance (unused).
        k: Integer neighbor count, 2 <= k < n_distinct. None resolves to 20,
            the same default as an omitted argument; requested counts never clamp.
        return_per_sample: Return an array in input row order instead of the mean.
        cache: Shared compute_knn cache. The conditioned cloud gets its own entry.
        dedup: If False, reject duplicate observations before distance computation.

    Raises:
        MeasurementUnavailable: Invalid parameters or evidence, precision loss
            merging distinct points, nonfinite/nonpositive distances, or an
            undefined estimate for any point. No observations are discarded to
            salvage a mean.

    Notes:
        The previous absolute radius clamp and log epsilon measured numerical
        constants at tiny scales or duplicate distances. Neither belongs in a
        scale-free distance ratio. Float32 conditioning can still leave geometry
        unresolved; such cases are refused instead of assigned a dimension.
        A finite-sample estimate below one is allowed: ambient dimensionality
        does not impose a lower bound on this expression.
    """
    if k is None:
        k = 20
    # The k-th neighbor contributes log(r_k / r_k) = 0. A negative denominator
    # needs at least one other term, hence k >= 2 (necessary, not sufficient).
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 2:
        raise MeasurementUnavailable("k must be a nonboolean integer >= 2")
    k = int(k)

    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    x = np.asarray(embeddings)
    if (
        x.ndim != 2 or 0 in x.shape
        or not np.issubdtype(x.dtype, np.number)
        or np.iscomplexobj(x)
    ):
        raise MeasurementUnavailable("embeddings must be a nonempty real numeric matrix")
    if not np.isfinite(x).all():
        raise MeasurementUnavailable("non-finite embeddings carry no geometry to measure")

    original_points, inverse = _distinct_rows(x)
    n_points = original_points.shape[0]
    if not dedup and n_points != x.shape[0]:
        raise MeasurementUnavailable("duplicate observations are unsupported with dedup=False")

    points = _rms_normalise(original_points).astype(np.float32)
    if _distinct_rows(points)[0].shape[0] != n_points:
        raise MeasurementUnavailable(
            "distinct input points collapse at float32 working precision; LID is unresolved"
        )
    # Self is excluded, so n_points distinct points supply n_points - 1 neighbors.
    if k >= n_points:
        raise MeasurementUnavailable(
            f"only {n_points} distinct points for k={k}: need at least k+1"
        )

    distances, _ = compute_knn(points, k=k, include_self=False, cache=cache)
    distances = np.asarray(distances, dtype=np.float64)
    if not np.all(np.isfinite(distances) & (distances > 0)):
        raise MeasurementUnavailable(
            "neighbor distances must be finite and positive; geometry is unresolved "
            "at float32 working precision"
        )

    # Float64 division prevents ratios of positive float32 distances underflowing
    # to zero before the log. For sorted positive distances d_j <= r_k, each log
    # is <= 0; a finite positive -k/sum requires a finite strictly negative sum.
    r_k = distances[:, -1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_sum = np.sum(np.log(distances / r_k[:, None]), axis=1)
    if not np.all(np.isfinite(log_sum) & (log_sum < 0)):
        raise MeasurementUnavailable("LID requires a finite strictly negative log-distance sum")
    with np.errstate(over="ignore", divide="ignore"):
        lid_values = -k / log_sum
    if not np.all(np.isfinite(lid_values) & (lid_values > 0)):
        raise MeasurementUnavailable("LID estimates are not finite and positive at working precision")
    lid_values = lid_values[inverse]

    mean_lid = float(np.mean(lid_values))
    if not np.isfinite(mean_lid):
        raise MeasurementUnavailable("mean LID is not finite at working precision")
    logger.info(
        f"LocalIntrinsicDimensionality: mean LID = {mean_lid:.3f} "
        f"({n_points}/{x.shape[0]} distinct points)"
    )
    return lid_values if return_per_sample else mean_lid
