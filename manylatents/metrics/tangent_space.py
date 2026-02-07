import logging
from typing import Optional, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import ColormapInfo
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


def TangentSpaceApproximation(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    variance_threshold: float = 0.95,
    return_per_sample: bool = False,
    _knn_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Union[float, Tuple[np.ndarray, ColormapInfo]]:
    """
    Estimate the local manifold dimension by approximating the tangent space via PCA.

    Args:
        embeddings:   (n_samples, n_features) array.
        dataset:      (unused) kept for Protocol compatibility.
        module:       (unused) kept for Protocol compatibility.
        n_neighbors:  how many neighbors to use for local PCA.
        variance_threshold: cumulative variance threshold to determine local dim.
        return_per_sample: if True, return per-sample dimensions with viz metadata.
        _knn_cache:   Optional (distances, indices) tuple from precomputed kNN.
                      Indices should be shape (n_samples, max_k+1) including self.

    Returns:
        float: Average local dimension (if return_per_sample=False)
        Tuple[np.ndarray, ColormapInfo]: Per-sample dimensions with visualization
            metadata (if return_per_sample=True). The ColormapInfo specifies
            categorical rendering with dynamic label generation.
    """
    if _knn_cache is not None:
        # Use precomputed kNN indices, slice to required k
        _, indices = _knn_cache
        indices = indices[:, :n_neighbors + 1]
    else:
        _, indices = compute_knn(embeddings, k=n_neighbors, include_self=True)

    # Vectorized: gather neighborhoods, batch SVD in chunks to control memory
    n_samples = embeddings.shape[0]
    k = indices.shape[1] - 1  # exclude self
    chunk_size = max(1, min(10_000, int(2e9 / (k * embeddings.shape[1] * 4))))

    dims_array = np.empty(n_samples, dtype=np.int32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[indices[start:end, 1:]]  # (chunk, k, d)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        s = np.linalg.svd(centered, compute_uv=False)  # (chunk, min(k,d))
        s2 = s * s
        total_var = s2.sum(axis=1, keepdims=True)
        total_var = np.where(total_var > 0, total_var, 1.0)  # avoid div by zero
        var_explained = s2 / total_var
        cum_var = np.cumsum(var_explained, axis=1)
        # searchsorted per row: find first index where cum_var >= threshold
        local_dims = (cum_var < variance_threshold).sum(axis=1) + 1
        dims_array[start:end] = local_dims

    if return_per_sample:
        logger.info(f"TangentSpaceApproximation: Per-sample local dimensions: {dims_array}")

        viz_info = ColormapInfo(
            cmap="categorical",
            label_names=None,
            label_format="Dim = {}",
            is_categorical=True,
        )
        return dims_array, viz_info
    else:
        avg_dim = float(np.mean(dims_array))
        logger.info(f"TangentSpaceApproximation: Average local dimension computed as {avg_dim}")
        return avg_dim
