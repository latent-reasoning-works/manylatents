import logging
from typing import Optional, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


def _gram_batch(P: np.ndarray) -> np.ndarray:
    """Double-centered squared-distance (classical-MDS) Gram matrices for a batch of point sets.

    Args:
        P: (batch, m, d) array of m points per neighborhood.

    Returns:
        (batch, m, m) array B = -0.5 * J D2 J with J = I - 1/m, D2 the squared distances.
    """
    G = P @ np.swapaxes(P, -1, -2)
    dg = np.einsum("...ii->...i", G)
    D2 = np.maximum(dg[..., :, None] + dg[..., None, :] - 2.0 * G, 0.0)
    m = D2.shape[-1]
    J = np.eye(m, dtype=D2.dtype) - 1.0 / m
    return -0.5 * (J @ D2 @ J)


@register_metric(
    aliases=["local_strain", "strain"],
    default_params={"n_neighbors": 25, "return_per_sample": False},
    description="Localized classical-MDS strain between input and embedded neighborhoods",
)
def LocalStrain(
    embeddings: np.ndarray,
    dataset,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    include_self: bool = True,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
    chunk_size: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Per-point localized classical-MDS strain of an embedding.

    For each point, take its k-nearest-neighbor neighborhood in the HIGH-DIMENSIONAL data
    (by default the point itself plus its k neighbors, m = k + 1), form the double-centered
    squared-distance Gram matrices B_in (from the high-dimensional coordinates) and B_out
    (from the embedded coordinates), and report the relative Frobenius residual after the
    scale-optimal scalar
    s = <B_in, B_out> / <B_out, B_out>:

        strain_i = sqrt( ||B_in - s B_out||_F^2 / ||B_in||_F^2 )

    The optimal scalar makes the value invariant to the arbitrary global scale of embedding
    coordinates. strain_i = 0 means the neighborhood is embedded as an exact similarity
    transform of its high-dimensional configuration; strain_i near 1 means the embedded
    configuration carries none of the neighborhood's inner-product structure.

    This is the L2 / isometry error of the classical-MDS objective, complementary to the
    ordinal set-overlap error measured by KNNPreservation: strain is the quantity the
    Eckart-Young tail bound (see CovarianceFloor) lower-bounds.

    Parameters:
        embeddings: Low-dimensional embeddings of shape (n_samples, n_components).
        dataset: An object with an attribute 'data' (the high-dimensional data).
        module: (unused) kept for Protocol compatibility.
        n_neighbors: Number of neighbors (k) defining each local neighborhood.
        include_self: If True (default) the neighborhood is the point plus its k input-space
            neighbors (m = k + 1); if False it is the k neighbors alone. The default keeps the
            point in its own neighborhood, which is what the strain of a local configuration
            means. Note the companion CovarianceFloor metric defaults to include_self=False,
            since a local covariance is taken about the neighborhood mean.
        return_per_sample: If True, return per-sample strain values; else return the mean.
        cache: Optional shared cache dict. Passed through to compute_knn().
        chunk_size: Points per batch. Defaults to a memory-bounded choice.

    Returns:
        float: Mean strain (if return_per_sample=False).
        np.ndarray: Per-sample strain values (if return_per_sample=True).
    """
    X = np.asarray(dataset.data, dtype=np.float64)
    Y = np.asarray(embeddings, dtype=np.float64)
    n_samples = X.shape[0]

    _, idx = compute_knn(X, k=n_neighbors, include_self=False, cache=cache)
    if include_self:
        # neighborhood = self followed by the k input-space neighbors
        nb = np.concatenate([np.arange(n_samples, dtype=idx.dtype)[:, None], idx], axis=1)
    else:
        nb = idx

    m = nb.shape[1]
    if chunk_size is None:
        chunk_size = max(1, min(10_000, int(2e8 / (m * m * 8))))

    strain = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        sel = nb[start:end]
        B_in = _gram_batch(X[sel])
        B_out = _gram_batch(Y[sel])
        num = (B_in * B_out).sum(axis=(1, 2))
        den = (B_out * B_out).sum(axis=(1, 2)) + 1e-30
        s = (num / den)[:, None, None]
        resid = ((B_in - s * B_out) ** 2).sum(axis=(1, 2))
        norm = (B_in * B_in).sum(axis=(1, 2)) + 1e-30
        strain[start:end] = np.sqrt(resid / norm)

    if return_per_sample:
        logger.info(f"LocalStrain: per-sample strain, mean={strain.mean():.3f}")
        return strain

    mean_strain = float(np.mean(strain))
    logger.info(f"LocalStrain: mean local strain = {mean_strain:.3f}")
    return mean_strain
