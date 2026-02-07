import logging
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import ColormapInfo

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
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)

    dims = []
    for idx in indices:
        local_points = embeddings[idx[1:]]  # Exclude the point itself.
        local_centered = local_points - np.mean(local_points, axis=0)
        # Perform SVD on the local neighborhood.
        _, s, _ = np.linalg.svd(local_centered, full_matrices=False)
        variance_explained = (s ** 2) / np.sum(s ** 2)
        cum_variance = np.cumsum(variance_explained)
        local_dim = np.searchsorted(cum_variance, variance_threshold) + 1
        dims.append(local_dim)
    
    if return_per_sample:
        # Round to integers for categorical labeling
        dims_array = np.array(dims, dtype=int)
        logger.info(f"TangentSpaceApproximation: Per-sample local dimensions (unique: {np.unique(dims_array)})")

        # Provide visualization metadata - plotting util will generate colors dynamically
        viz_info = ColormapInfo(
            cmap="categorical",  # Signal to generate discrete colors
            label_names=None,  # Will be generated from data
            label_format="Dim = {}",  # Template for label generation
            is_categorical=True,
        )
        return dims_array, viz_info
    else:
        avg_dim = float(np.mean(dims))
        logger.info(f"TangentSpaceApproximation: Average local dimension computed as {avg_dim}")
        return avg_dim
