import warnings
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.utils.metrics import compute_eigenvalues


def affinity_spectrum(affinity_matrix: np.ndarray, top_k: int = 25) -> np.ndarray:
    """
    Compute the top `k` eigenvalues of a symmetric affinity matrix.

    This function expects a symmetric affinity matrix (e.g., from symmetric
    diffusion operator normalization). For symmetric matrices:
    - All eigenvalues are real (no complex values)
    - Eigenvalues of PSD matrices are non-negative (no negative values)

    Parameters:
        affinity_matrix (np.ndarray): Symmetric affinity matrix.
        top_k (int): Number of top eigenvalues to return.

    Returns:
        np.ndarray: Top `k` eigenvalues in descending order.
    """
    eigenvalues = np.linalg.eigvalsh(affinity_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    return eigenvalues_sorted[:top_k]


##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def AffinitySpectrum(dataset, embeddings: np.ndarray, module: LatentModule,
                     top_k: int = 25, cache: Optional[dict] = None) -> np.ndarray:
    """
    Compute affinity spectrum from the module's symmetric affinity matrix.

    Uses compute_eigenvalues with cache for shared eigendecomposition.

    Args:
        dataset: Dataset object (unused).
        embeddings: Low-dimensional embeddings (unused).
        module: LatentModule instance with affinity_matrix method.
        top_k: Number of top eigenvalues to return.
        cache: Shared cache dict. Pass through to compute_eigenvalues().

    Returns:
        Array of top_k eigenvalues, or [nan] if affinity matrix not available.
    """
    eigenvalues = compute_eigenvalues(module, cache=cache)
    if eigenvalues is not None:
        return eigenvalues[:top_k]

    # Fallback: legacy path if compute_eigenvalues returned None
    try:
        affinity_mat = module.affinity_matrix(use_symmetric=True)
    except (NotImplementedError, AttributeError, TypeError):
        warnings.warn(
            f"AffinitySpectrum metric skipped: {module.__class__.__name__} does not expose an affinity_matrix method.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return affinity_spectrum(affinity_matrix=affinity_mat, top_k=top_k)
