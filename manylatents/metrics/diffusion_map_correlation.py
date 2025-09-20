import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Literal

from manylatents.algorithms.latent_module_base import LatentModule
from manylatents.algorithms.diffusionmap_algorithm import compute_dm


def diffusion_map_correlation(
    embeddings: np.ndarray,
    kernel_matrix: np.ndarray,
    dm_components: int = 2,
    alpha: float = 1.0,
    correlation_type: Literal["pearson", "spearman"] = "pearson"
) -> np.ndarray:
    """
    Compute correlation between embedding coordinates and corresponding diffusion map coordinates.

    Parameters:
    -----------
    embeddings : np.ndarray
        Low-dimensional embeddings (n_samples x n_dims)
    kernel_matrix : np.ndarray
        Kernel/affinity matrix from the dimensionality reduction algorithm
    dm_components : int, default=2
        Number of diffusion map components to use (typically 2 for DM1, DM2)
    alpha : float, default=1.0
        Diffusion map parameter (0=Graph Laplacian, 0.5=Fokker-Plank, 1=Laplace-Beltrami)
    correlation_type : {"pearson", "spearman"}
        Type of correlation to compute

    Returns:
    --------
    np.ndarray
        Array of correlations: [corr(embed1, DM1), corr(embed2, DM2), ...]
        One correlation score per diffusion map coordinate
    """
    # Take only the first dm_components dimensions of embeddings
    n_components = min(embeddings.shape[1], dm_components)
    embeddings_subset = embeddings[:, :n_components]

    # Compute diffusion map coordinates
    evecs_right, evals, _, _ = compute_dm(kernel_matrix, alpha=alpha)

    # Get diffusion map coordinates (skip first eigenvector which is constant)
    # Use t=1 for standard diffusion coordinates
    dm_coords = evecs_right[:, 1:(n_components+1)] @ np.diag(evals[1:(n_components+1)])

    # Compute correlations: embedding dim i vs diffusion map dim i
    correlations = []

    for i in range(n_components):
        if correlation_type == "pearson":
            corr, _ = pearsonr(embeddings_subset[:, i], dm_coords[:, i])
        elif correlation_type == "spearman":
            corr, _ = spearmanr(embeddings_subset[:, i], dm_coords[:, i])
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")

        correlations.append(abs(corr))  # Take absolute value - sign doesn't matter

    return np.array(correlations)


##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def DiffusionMapCorrelation(
    dataset,
    embeddings: np.ndarray,
    module: LatentModule,
    dm_components: int = 2,
    alpha: float = 1.0,
    correlation_type: Literal["pearson", "spearman"] = "pearson"
) -> np.ndarray:
    """
    Compute correlation between embeddings and corresponding diffusion map coordinates.

    This metric measures how well each embedding dimension (e.g., PHATE1, PHATE2) correlates
    with its corresponding diffusion map coordinate (DM1, DM2).

    Parameters:
    -----------
    dataset : object
        Dataset object (not used but required for Metric protocol)
    embeddings : np.ndarray
        Low-dimensional embeddings (n_samples x n_dims)
    module : LatentModule
        Module containing kernel_matrix attribute
    dm_components : int, default=2
        Number of diffusion map components to use
    alpha : float, default=1.0
        Diffusion map normalization parameter
    correlation_type : {"pearson", "spearman"}
        Type of correlation to compute

    Returns:
    --------
    np.ndarray
        Array of correlations: [corr(embed1, DM1), corr(embed2, DM2), ...]
        One correlation score per diffusion map coordinate
    """
    kernel_matrix = getattr(module, "kernel_matrix", None)
    if kernel_matrix is None:
        warnings.warn(
            "DiffusionMapCorrelation metric skipped: module has no 'kernel_matrix' attribute.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return diffusion_map_correlation(
        embeddings=embeddings,
        kernel_matrix=kernel_matrix,
        dm_components=dm_components,
        alpha=alpha,
        correlation_type=correlation_type
    )