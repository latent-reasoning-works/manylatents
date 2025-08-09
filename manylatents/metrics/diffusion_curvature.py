import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Optional

def diffusionCurvature(embedding: np.ndarray, t: int = 3, alpha: float = 1.0, percentile: float = 5) -> np.ndarray:
    """
    Compute pointwise diffusion curvature on a given embedding.

    Parameters:
    - embedding: np.ndarray of shape (n_samples, n_dims)
    - t: number of diffusion steps
    - alpha: kernel normalization exponent
    - percentile: defines radius r for B(x, r)

    Returns:
    - C: np.ndarray of shape (n_samples,) representing curvature at each point
    """
    D = pairwise_distances(embedding, metric="euclidean")
    sigma = np.median(D**2)
    G = np.exp(-D**2 / sigma)

    q = G.sum(axis=1)
    K = G / (q[:, None]**alpha * q[None, :]**alpha)

    P = K / K.sum(axis=1, keepdims=True)
    P_t = np.linalg.matrix_power(P, t)

    D_diff = pairwise_distances(P_t, metric="euclidean")
    r = np.percentile(D_diff, percentile)

    balls = [np.where(D_diff[i] <= r)[0] for i in range(len(embedding))]

    C = np.array([
        P_t[i, balls[i]].sum() / len(balls[i]) if len(balls[i]) > 0 else 0
        for i in range(len(embedding))
    ])
    
    return C

def DiffusionCurvature(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    t: int = 3,
    alpha: float = 1.0,
    percentile: float = 5
) -> np.ndarray:
    """
    Wrapper for diffusionCurvature.
    """
    return diffusionCurvature(embeddings, t=t, alpha=alpha, percentile=percentile)

