import warnings
import numpy as np
from manylatents.algorithms.latent_module_base import LatentModule


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
    # Compute eigenvalues of symmetric matrix
    # For symmetric matrices, eigenvalues are always real
    eigenvalues = np.linalg.eigvalsh(affinity_matrix)

    # Sort in descending order and return top-k
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    return eigenvalues_sorted[:top_k]

##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def AffinitySpectrum(dataset, embeddings: np.ndarray, module: LatentModule, top_k: int = 25) -> np.ndarray:
    """
    Compute affinity spectrum from the module's symmetric affinity matrix.

    This metric requests the symmetric version of the affinity matrix to guarantee
    real, non-negative eigenvalues. The symmetric normalization preserves the
    spectral properties of the underlying kernel while ensuring numerical stability.

    Args:
        dataset: Dataset object (unused).
        embeddings: Low-dimensional embeddings (unused).
        module: LatentModule instance with affinity_matrix method.
        top_k: Number of top eigenvalues to return.

    Returns:
        Array of top_k eigenvalues, or [nan] if affinity matrix not available.
    """
    try:
        # Request symmetric affinity matrix for positive eigenvalue guarantee
        affinity_mat = module.affinity_matrix(use_symmetric=True)
    except (NotImplementedError, AttributeError, TypeError):
        warnings.warn(
            f"AffinitySpectrum metric skipped: {module.__class__.__name__} does not expose an affinity_matrix method.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return affinity_spectrum(affinity_matrix=affinity_mat, top_k=top_k)
