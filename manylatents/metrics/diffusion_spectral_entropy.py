import numpy as np
from sklearn.metrics import pairwise_distances
import random
from typing import Optional

from manylatents.metrics.registry import register_metric
from manylatents.utils.kernel_utils import symmetric_diffusion_operator


def exact_eigvals(K: np.ndarray) -> np.ndarray:
    """Compute exact eigenvalues with symmetry safety check.

    Uses eigvalsh for symmetric matrices (more stable, real eigenvalues)
    and eigvals for general matrices.
    """
    if np.allclose(K, K.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix - use eigvalsh (faster, guaranteed real)
        return np.linalg.eigvalsh(K)
    else:
        # General matrix - use eigvals
        return np.linalg.eigvals(K)


def approx_eigvals(K: np.ndarray, exact_threshold: int = 1000) -> np.ndarray:
    """
    Approximate eigenvalues using sparse eigensolver for large matrices.

    Parameters:
        K: Input matrix
        exact_threshold: Use exact eigvals for matrices smaller than this (default 1000)

    For matrices above threshold, uses scipy.sparse.linalg.eigsh for top-k eigenvalues.
    """
    if K.shape[0] < exact_threshold:
        return exact_eigvals(K)

    try:
        from scipy.sparse.linalg import eigsh
        k = min(100, K.shape[0] - 2)
        return eigsh(K, k=k, which='LM', return_eigenvectors=False)
    except Exception:
        return exact_eigvals(K)


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0, alpha: float = 0.5):
    '''
    Given input X returns a symmetric diffusion operator S with the same
    eigenvalues as the row-stochastic diffusion matrix P.

    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Inputs:
        X: a numpy array of size n x d
        sigma: bandwidth of the Gaussian kernel
        alpha: density normalization power (0=graph Laplacian, 0.5=Fokker-Planck, 1=Laplace-Beltrami)

    Returns:
        S: symmetric (n, n) matrix with same eigenvalues as the diffusion matrix P.
    '''
    D = pairwise_distances(X)
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))
    S = symmetric_diffusion_operator(G, alpha=alpha)
    return S


def compute_diffusion_matrix_knn(
    distances: np.ndarray,
    indices: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Build diffusion operator from kNN graph with adaptive bandwidth.

    Uses Zelnik-Manor & Perona (2004) adaptive bandwidth: sigma_i = distance
    to the k-th neighbor. Builds a sparse Gaussian kernel, symmetrizes via
    union of neighborhoods, then normalizes with symmetric_diffusion_operator.

    Args:
        distances: (n, k+1) from compute_knn, self at index 0.
        indices: (n, k+1) from compute_knn, self at index 0.
        alpha: Density normalization power.
            0 = graph Laplacian, 0.5 = Fokker-Planck, 1.0 = Laplace-Beltrami.

    Returns:
        (n, n) symmetric diffusion operator (dense, for eigendecomposition).
    """
    n = distances.shape[0]

    # Adaptive bandwidth: distance to k-th neighbor (last column)
    sigma = distances[:, -1].copy()
    sigma[sigma == 0] = 1e-10  # guard against zero distance

    # Build sparse Gaussian kernel from kNN edges
    K = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(1, distances.shape[1]):  # skip self at index 0
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            K[i, j] = np.exp(-(d ** 2) / (sigma[i] * sigma[j]))

    # Symmetrize: union of neighborhoods
    K = np.maximum(K, K.T)

    # Fill diagonal (self-affinity = 1)
    np.fill_diagonal(K, 1.0)

    # Apply alpha-normalization via shared utility
    S = symmetric_diffusion_operator(K, alpha=alpha)

    return S


@register_metric(
    aliases=["diffusion_spectral_entropy", "dse"],
    default_params={"t": 3, "gaussian_kernel_sigma": 10, "output_mode": "entropy", "t_high": 100, "numerical_floor": 1e-6, "max_N": 10000, "random_seed": 0, "kernel": "knn", "k": 15, "alpha": 1.0},
    description="Diffusion spectral entropy (eigenvalue count at diffusion time t)",
)
def DiffusionSpectralEntropy(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    t: int = 3,
    gaussian_kernel_sigma: float = 10,
    output_mode: str = "entropy",
    t_high: int = 100,
    numerical_floor: float = 1e-6,
    max_N: int = 10000,
    random_seed: int = 0,
    cache: Optional[dict] = None,
    kernel: str = "knn",
    k: int = 15,
    alpha: float = 1.0,
) -> float:
    """
    Diffusion spectral entropy and eigenvalue counting on embedding data.

    Builds a diffusion operator from the embeddings, computes its eigenvalues,
    and returns either entropy or eigenvalue counts at a given diffusion time.

    Parameters:
        embeddings: Input data array (n_samples, n_features)
        dataset: Provided for protocol compliance (unused)
        module: Provided for protocol compliance (unused)
        t: Diffusion time for entropy mode
        gaussian_kernel_sigma: Bandwidth of Gaussian kernel (dense mode only)
        output_mode: "entropy", "eigenvalue_count", "eigenvalue_count_full", "eigenvalue_count_sweep"
        t_high: High diffusion time for asymptotic eigenvalue counting
        numerical_floor: Noise floor for eigenvalue counting
        max_N: Max samples (subsample if larger)
        random_seed: Seed for reproducible subsampling
        cache: Shared cache dict for kNN reuse
        kernel: "knn" (adaptive bandwidth, default) or "dense" (global Gaussian)
        k: Neighborhood size for knn kernel (default 15)
        alpha: Density normalization (0=graph Laplacian, 0.5=Fokker-Planck, 1.0=Laplace-Beltrami)
    """
    # Subsample if too large
    X = embeddings
    if max_N is not None and len(X) > max_N:
        random.seed(random_seed)
        rand_inds = np.array(random.sample(range(len(X)), k=max_N))
        X = X[rand_inds, :]

    # Dispatch kernel construction
    if kernel == "knn":
        from manylatents.utils.metrics import compute_knn
        distances, indices = compute_knn(X, k=k, cache=cache)
        K = compute_diffusion_matrix_knn(distances, indices, alpha=alpha)
    else:
        K = compute_diffusion_matrix(X, sigma=gaussian_kernel_sigma, alpha=alpha)

    if output_mode in ("eigenvalue_count", "eigenvalue_count_full", "eigenvalue_count_sweep"):
        eigvals = exact_eigvals(K)
        eigvals = np.abs(eigvals)

        if output_mode == "eigenvalue_count_sweep":
            t_values = t_high if isinstance(t_high, (list, tuple)) else [t_high]
            results = {"raw_eigvals": eigvals}
            for t_val in t_values:
                eigvals_powered = eigvals ** t_val
                count = float(np.sum(eigvals_powered > numerical_floor))
                results[f"count_t{t_val}"] = count
                results[f"spectrum_t{t_val}"] = eigvals_powered
            return results

        eigvals_powered = eigvals ** t_high
        count = float(np.sum(eigvals_powered > numerical_floor))

        if output_mode == "eigenvalue_count_full":
            return {
                "count": count,
                "spectrum": eigvals_powered,
                "above_floor": eigvals_powered[eigvals_powered > numerical_floor],
            }
        return count

    # Default: entropy mode
    eigvals = exact_eigvals(K)
    eigvals = np.abs(eigvals)
    eigvals = eigvals ** t
    prob = eigvals / eigvals.sum()
    prob = prob + np.finfo(float).eps
    entropy = -np.sum(prob * np.log2(prob))
    return entropy
