import logging
from typing import Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def TangentSpaceApproximation(dataset, 
                              embeddings: np.ndarray, 
                              n_neighbors: int = 10, 
                              variance_threshold: float = 0.95, 
                              return_per_sample: bool = False) -> Union[float, np.ndarray]:
    """
    Estimate the local manifold dimension by approximating the tangent space via PCA.

    If return_per_sample is False (default), returns the average local dimension (float).
    If return_per_sample is True, returns an array of local dimensions, one per observation.
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        logger.info(f"TangentSpaceApproximation: Converted embeddings to numpy with shape {embeddings.shape}")

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
        dims_array = np.array(dims)
        logger.info(f"TangentSpaceApproximation: Per-sample local dimensions: {dims_array}")
        return dims_array
    else:
        avg_dim = float(np.mean(dims))
        logger.info(f"TangentSpaceApproximation: Average local dimension computed as {avg_dim}")
        return avg_dim
