import numpy as np


def symmetric_diffusion_operator(kernel_matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Compute symmetric diffusion operator from kernel matrix.

    Applies symmetric normalization D^{-1/2} K D^{-1/2} which preserves symmetry
    and ensures all eigenvalues are real and non-negative for PSD kernels.

    This normalization differs from row-stochastic (D^{-1} K) which produces an
    asymmetric matrix that can have negative or complex eigenvalues.

    Parameters:
        kernel_matrix (np.ndarray): Symmetric kernel/affinity matrix.
        alpha (float): Power for degree normalization. Default 1.0.

    Returns:
        np.ndarray: Symmetric normalized affinity matrix (diffusion operator).
    """
    K = kernel_matrix

    # Step 1: Degree normalization
    d = np.power(K.sum(axis=1), alpha)
    D_inv = np.diag(1 / d)

    # Step 2: Normalize kernel
    K_alpha = D_inv @ K @ D_inv

    # Step 3: Build diffusion operator (symmetric normalization)
    # This D^{-1/2} K D^{-1/2} form preserves symmetry and ensures positive eigenvalues
    d_alpha = K_alpha.sum(axis=1)
    D_sqrt_inv_alpha = np.diag(1 / np.sqrt(d_alpha))
    S = D_sqrt_inv_alpha @ K_alpha @ D_sqrt_inv_alpha

    return S
