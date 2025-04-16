import numpy as np
from scipy.linalg import svd
from src.algorithms.dimensionality_reduction import DimensionalityReductionModule

def affinity_spectrum(embeddings: np.ndarray, kernel_matrix: np.ndarray, top_k: int = 25) -> np.ndarray:
    """
    Compute the top `k` singular values of the normalized affinity matrix
    derived from a kernel matrix, using the symmetrized diffusion operator.

    Parameters:
        embeddings (np.ndarray): Low-dimensional embeddings (currently unused).
        kernel_matrix (np.ndarray): Kernel or affinity matrix.
        top_k (int): Number of top singular values to return.

    Returns:
        np.ndarray: Top `k` singular values of the diffusion operator.
    """
    K = kernel_matrix
    alpha = 1.0

    # Step 1: Degree normalization
    d = np.power(K.sum(axis=1), alpha)
    D_inv = np.diag(1 / d)

    # Step 2: Normalize kernel
    K_alpha = D_inv @ K @ D_inv

    # Step 3: Build diffusion operator (symmetric normalization)
    d_alpha = K_alpha.sum(axis=1)
    D_sqrt_inv_alpha = np.diag(1 / np.sqrt(d_alpha))
    S = D_sqrt_inv_alpha @ K_alpha @ D_sqrt_inv_alpha

    # Step 4: SVD (returns U, singular values, V^T)
    U, svals, VT = svd(S)

    return svals[:top_k]

##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def AffinitySpectrum(dataset, embeddings: np.ndarray, module: DimensionalityReductionModule, top_k: int = 25) -> np.ndarray:
    return affinity_spectrum(embeddings=embeddings, top_k=top_k, kernel_matrix=module.kernel_matrix)
