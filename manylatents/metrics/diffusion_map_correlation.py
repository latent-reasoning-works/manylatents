import warnings
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Literal, Optional

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.diffusion_map import compute_dm
from manylatents.metrics.registry import register_metric


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
    evecs_right, evals, _, _, _ = compute_dm(kernel_matrix, alpha=alpha)

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

@register_metric(
    aliases=["diffusion_map_correlation"],
    default_params={"dm_components": 2, "alpha": 1.0, "correlation_type": "pearson"},
    description="Correlation between diffusion map and embedding distances",
)
def DiffusionMapCorrelation(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    dm_components: int = 2,
    alpha: float = 1.0,
    correlation_type: Literal["pearson", "spearman"] = "pearson"
) -> np.ndarray:
    """
    Compute correlation between embeddings and corresponding diffusion map coordinates.

    This metric measures how well each embedding dimension (e.g., PHATE1, PHATE2)
    correlates with its corresponding diffusion map coordinate (DM1, DM2).

    Parameters
    ----------
    embeddings : np.ndarray
        Low-dimensional embeddings of shape (n_samples, n_dims).
    dataset : object, optional
        Dataset object (not used directly, but included for Metric protocol compatibility).
    module : LatentModule, optional
        Module expected to contain a `kernel_matrix` attribute used to compute
        diffusion map coordinates.
    dm_components : int, default=2
        Number of diffusion map components to use for correlation.
    alpha : float, default=1.0
        Diffusion map normalization parameter:
        - 0   → Graph Laplacian
        - 0.5 → Fokker–Planck
        - 1   → Laplace–Beltrami
    correlation_type : {"pearson", "spearman"}, default="pearson"
        Correlation type to compute between embedding axes and diffusion map axes.

    Returns
    -------
    np.ndarray
        Array of absolute correlation values:
        [corr(embed1, DM1), corr(embed2, DM2), ...].
        One correlation score per diffusion map coordinate.
    """
    # Skip for modules that don't have kernel_matrix
    if module is None or not hasattr(module, "kernel_matrix"):
        warnings.warn(
            "DiffusionMapCorrelation metric skipped: missing module.kernel_matrix.",
            RuntimeWarning
        )
        return float("nan")

    # Skip for specific algorithm types where this metric is not appropriate
    module_name = module.__class__.__name__
    skip_modules = {
        "PCAModule": "Gram matrix not appropriate as kernel matrix",
        "MDSModule": "Distance-based method without appropriate kernel matrix",
        "DiffusionMapModule": "Would be trivially 1.0 (diffusion maps vs diffusion maps)"
    }

    if module_name in skip_modules:
        warnings.warn(
            f"DiffusionMapCorrelation metric skipped for {module_name}: {skip_modules[module_name]}",
            RuntimeWarning
        )
        return float("nan")

    try:
        kernel_matrix = module.kernel_matrix()
    except (NotImplementedError, AttributeError):
        warnings.warn(
            f"DiffusionMapCorrelation metric skipped: {module_name} does not expose a kernel_matrix.",
            RuntimeWarning
        )
        return float("nan")

    return diffusion_map_correlation(
        embeddings=embeddings,
        kernel_matrix=kernel_matrix,
        dm_components=dm_components,
        alpha=alpha,
        correlation_type=correlation_type
    )
