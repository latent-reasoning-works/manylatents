import logging
from typing import Optional, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


def _local_eigenvalues(
    data: np.ndarray,
    n_neighbors: int,
    include_self: bool,
    cache: Optional[dict],
) -> np.ndarray:
    """Eigenvalues of every point's local centered Gram / covariance, descending.

    Args:
        data: (n_samples, n_features) array.
        n_neighbors: Number of neighbors (k) per neighborhood.
        include_self: If True the point joins its own neighborhood (m = k + 1).
        cache: Optional shared cache dict passed to compute_knn().

    Returns:
        (n_samples, min(m, n_features)) array of eigenvalues lambda_i = s_i^2, where s_i are
        the singular values of the mean-centered neighborhood coordinate matrix.
    """
    n_samples = data.shape[0]
    _, idx = compute_knn(data, k=n_neighbors, include_self=False, cache=cache)
    if include_self:
        idx = np.concatenate([np.arange(n_samples, dtype=idx.dtype)[:, None], idx], axis=1)

    m = idx.shape[1]
    chunk_size = max(1, min(10_000, int(2e9 / (m * data.shape[1] * 4))))

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = data[idx[start:end]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

    s = np.concatenate(sv_chunks, axis=0)
    return s.astype(np.float64) ** 2


@register_metric(
    aliases=["covariance_floor", "strain_floor", "phi_floor"],
    default_params={"n_neighbors": 25, "target_dim": 2, "return_per_sample": False},
    description="Relative strain floor Phi(d): discarded tail of the local covariance spectrum",
)
def CovarianceFloor(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    target_dim: int = 2,
    include_self: bool = False,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray]:
    """
    Per-point relative strain floor Phi(d) of the local covariance spectrum.

    For each point, take the eigenvalues lambda_1 >= lambda_2 >= ... of the centered Gram /
    covariance of its k-nearest-neighbor neighborhood and report the relative mass carried by
    the discarded tail beyond the first d of them:

        Phi(d) = sum_{i > d} lambda_i^2 / sum_i lambda_i^2

    By Eckart-Young-Mirsky this is the relative squared Frobenius error of the best rank-d
    approximation of the local Gram matrix, so it is an a-priori (input-only) lower bound on
    the strain any d-dimensional embedding of that neighborhood can achieve -- the quantity
    LocalStrain measures on a realized embedding. Phi(d) lies in [0, 1]; Phi(d) = 0 means the
    neighborhood is exactly d-dimensional, and larger values mean more of the local
    second-order structure cannot survive the projection to d dimensions.

    The eigenvalues are SQUARED in both numerator and denominator: the bound is on the
    Frobenius error of the Gram matrix, whose singular values are the covariance eigenvalues.
    The unsquared ratio sum_{i>d} lambda_i / sum_i lambda_i is the familiar discarded-variance
    ratio, a different quantity.

    Parameters:
        embeddings: (n_samples, n_features) array whose local geometry is analyzed. This is
            usually the high-dimensional INPUT data (the floor is an input-only quantity);
            pass it via the `dataset` metric context.
        dataset: (unused) kept for Protocol compatibility.
        module: (unused) kept for Protocol compatibility.
        n_neighbors: Number of neighbors (k) defining each local neighborhood.
        target_dim: The embedding dimension d whose floor is reported (2 for a 2D embedding).
        include_self: If False (default) the neighborhood is the k neighbors alone, matching
            the local-covariance convention shared with LocalSpectralAnalysis; if True the
            point joins its own neighborhood. (The companion LocalStrain metric defaults to
            include_self=True, since the strain of a configuration includes the point.)
        return_per_sample: If True, return per-sample Phi(d); else return the mean.
        cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
        float: Mean Phi(d) (if return_per_sample=False).
        np.ndarray: Per-sample Phi(d) values (if return_per_sample=True).
    """
    lam = _local_eigenvalues(np.asarray(embeddings), n_neighbors, include_self, cache)
    lam2 = lam ** 2
    total = lam2.sum(axis=1) + 1e-30
    tail = lam2[:, target_dim:].sum(axis=1) if lam2.shape[1] > target_dim else np.zeros(len(lam2))
    phi = tail / total

    if return_per_sample:
        logger.info(f"CovarianceFloor: per-sample Phi({target_dim}), mean={phi.mean():.4f}")
        return phi

    mean_phi = float(np.mean(phi))
    logger.info(f"CovarianceFloor: mean Phi({target_dim}) = {mean_phi:.4f}")
    return mean_phi
