import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Optional
import warnings
from manylatents.algorithms.latent.latent_module_base import LatentModule

def diffusion_curvature(P: np.ndarray, t: int = 3, percentile: float = 5) -> np.ndarray:
    """
    Compute pointwise diffusion curvature from a transition matrix.

    Parameters:
    - P: np.ndarray of shape (n_samples, n_samples), row-stochastic transition matrix
    - t: number of diffusion steps
    - percentile: defines radius r for B(x, r) in diffusion space

    Returns:
    - C: np.ndarray of shape (n_samples,) representing curvature at each point
    """
    # Compute t-step diffusion
    P_t = np.linalg.matrix_power(P, t)

    # Compute pairwise distances in diffusion space
    D_diff = pairwise_distances(P_t, metric="euclidean")
    r = np.percentile(D_diff, percentile)

    # Find balls in diffusion space
    balls = [np.where(D_diff[i] <= r)[0] for i in range(len(P))]

    # Compute curvature: average diffusion probability within local ball
    C = np.array([
        P_t[i, balls[i]].sum() / len(balls[i]) if len(balls[i]) > 0 else 0
        for i in range(len(P))
    ])

    return C

##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def DiffusionCurvature(
    dataset,
    embeddings: np.ndarray,
    module: LatentModule,
    t: int = 3,
    percentile: float = 5
) -> np.ndarray:
    """
    Compute diffusion curvature from the module's affinity matrix.

    Uses the module's transition matrix to compute pointwise diffusion curvature,
    which measures how diffusion probability concentrates in local neighborhoods
    after t diffusion steps.

    Args:
        dataset: Dataset object (unused).
        embeddings: Low-dimensional embeddings (unused).
        module: LatentModule instance with affinity_matrix method.
        t: Number of diffusion steps.
        percentile: Percentile for defining local ball radius in diffusion space.

    Returns:
        Array of per-sample curvature values, or [nan] if affinity matrix not available.
    """
    try:
        # Get row-stochastic transition matrix (1-step)
        P = module.affinity_matrix(use_symmetric=False)
    except (NotImplementedError, AttributeError, TypeError):
        warnings.warn(
            f"DiffusionCurvature metric skipped: {module.__class__.__name__} does not expose an affinity_matrix method.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return diffusion_curvature(P=P, t=t, percentile=percentile)

