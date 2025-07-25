from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from src.algorithms.latent_module_base import LatentModule


def Continuity(embeddings: np.ndarray,
               dataset: Optional[object] = None,
               module: Optional[LatentModule] = None,
               n_neighbors: int = 25,
               metric: str = 'euclidean',
               return_per_sample: bool = False,
               normalize: bool = True,
               adjust_for_random: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute (normalized and optionally adjusted) continuity between high-D and embedded space.

    Parameters:
    - embeddings: low-dimensional embedding (n_samples, d)
    - dataset: object with .data attribute (high-dimensional data, n_samples, D)
    - n_neighbors: size of the neighborhood (K')
    - metric: distance metric
    - return_per_sample: if True, also return pointwise overlap values per sample
    - normalize: if True, apply Equation (9) normalization (divide by n_neighbors)
    - adjust_for_random: if True, apply Equation (10) adjustment for expected random overlap

    Returns:
    - continuity_score: mean continuity (scalar)
    - (optional) pointwise_continuity: array of shape (n_samples,)
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        
    X_high = dataset.data
    X_low = embeddings
    n = X_high.shape[0]

    # Compute pairwise distances
    dist_high = pairwise_distances(X_high, metric=metric)
    dist_low = pairwise_distances(X_low, metric=metric)

    # Exclude self from neighborhoods
    np.fill_diagonal(dist_high, np.inf)
    np.fill_diagonal(dist_low, np.inf)

    # Get indices of k nearest neighbors in both spaces
    knn_high = np.argsort(dist_high, axis=1)[:, :n_neighbors]
    knn_low = np.argsort(dist_low, axis=1)[:, :n_neighbors]

    # Compute pointwise neighborhood overlaps
    pointwise_overlap = np.array([
        len(np.intersect1d(knn_high[i], knn_low[i], assume_unique=True))
        for i in range(n)
    ])

    # Normalize to [0, 1]
    if normalize:
        pointwise_continuity = pointwise_overlap / n_neighbors
    else:
        pointwise_continuity = pointwise_overlap.astype(float)

    # Adjust for expected random overlap
    if adjust_for_random:
        pointwise_continuity -= n_neighbors / (n - 1)

    # Mean continuity score
    continuity_score = float(np.mean(pointwise_continuity))

    if return_per_sample:
        return continuity_score, pointwise_continuity
    else:
        return continuity_score
