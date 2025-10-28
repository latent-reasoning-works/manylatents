import warnings
import numpy as np
from manylatents.algorithms.latent.latent_module_base import LatentModule


def kernel_matrix_sparsity(embeddings: np.ndarray, kernel_matrix: np.ndarray, threshold: float = 1e-10) -> float:
    """
    Compute the sparsity of the kernel matrix as the fraction of near-zero elements.

    This metric provides an easy way to compare different neighborhood sizes across
    dimensionality reduction algorithms. Higher sparsity indicates smaller effective
    neighborhood sizes, while lower sparsity indicates larger neighborhoods.

    Parameters:
        embeddings (np.ndarray): Low-dimensional embeddings (currently unused).
        kernel_matrix (np.ndarray): Kernel or affinity matrix from the LatentModule.
        threshold (float): Values below this threshold are considered zero. Default: 1e-10.

    Returns:
        float: Sparsity ratio between 0 and 1, where:
               - 0.0 = completely dense (no zeros)
               - 1.0 = completely sparse (all zeros)
    """
    if kernel_matrix.size == 0:
        return np.nan

    # Count elements below threshold (considered "zero")
    zero_elements = np.sum(np.abs(kernel_matrix) <= threshold)
    total_elements = kernel_matrix.size

    # Calculate sparsity as fraction of zero elements
    sparsity = zero_elements / total_elements

    return float(sparsity)


def kernel_matrix_density(embeddings: np.ndarray, kernel_matrix: np.ndarray, threshold: float = 1e-10) -> float:
    """
    Compute the density of the kernel matrix as the fraction of non-zero elements.

    This is the complement of sparsity: density = 1 - sparsity.

    Parameters:
        embeddings (np.ndarray): Low-dimensional embeddings (currently unused).
        kernel_matrix (np.ndarray): Kernel or affinity matrix from LatentModule.
        threshold (float): Values below this threshold are considered zero. Default: 1e-10.

    Returns:
        float: Density ratio between 0 and 1, where:
               - 0.0 = completely sparse (all zeros)
               - 1.0 = completely dense (no zeros)
    """
    sparsity = kernel_matrix_sparsity(embeddings, kernel_matrix, threshold)
    if np.isnan(sparsity):
        return np.nan
    return 1.0 - sparsity


##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def KernelMatrixSparsity(dataset, embeddings: np.ndarray, module: LatentModule, threshold: float = 1e-10) -> float:
    """
    Metric wrapper for kernel matrix sparsity computation.

    Parameters:
        dataset: Dataset object (unused but required by metric protocol).
        embeddings (np.ndarray): Low-dimensional embeddings from LatentModule.
        module (LatentModule): The fitted LatentModule.
        threshold (float): Values below this threshold are considered zero.

    Returns:
        float: Sparsity ratio or NaN if kernel_matrix is not available.
    """
    kernel_matrix = getattr(module, "kernel_matrix", None)
    if kernel_matrix is None:
        warnings.warn(
            "KernelMatrixSparsity metric skipped: module has no 'kernel_matrix' attribute.",
            RuntimeWarning
        )
        return np.nan

    return kernel_matrix_sparsity(embeddings=embeddings, kernel_matrix=kernel_matrix, threshold=threshold)


def KernelMatrixDensity(dataset, embeddings: np.ndarray, module: LatentModule, threshold: float = 1e-10) -> float:
    """
    Metric wrapper for kernel matrix density computation.

    Parameters:
        dataset: Dataset object (unused but required by metric protocol).
        embeddings (np.ndarray): Low-dimensional embeddings from LatentModule.
        module (LatentModule): The fitted LatentModule.
        threshold (float): Values below this threshold are considered zero.

    Returns:
        float: Density ratio or NaN if kernel_matrix is not available.
    """
    kernel_matrix = getattr(module, "kernel_matrix", None)
    if kernel_matrix is None:
        warnings.warn(
            "KernelMatrixDensity metric skipped: module has no 'kernel_matrix' attribute.",
            RuntimeWarning
        )
        return np.nan

    return kernel_matrix_density(embeddings=embeddings, kernel_matrix=kernel_matrix, threshold=threshold)