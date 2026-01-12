import numpy as np
from sklearn.metrics import pairwise_distances
import os
import random
from typing import Optional


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


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0):
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        K: a numpy array of size n x n that has the same eigenvalues as the diffusion matrix.
    '''

    # Construct the distance matrix.
    D = pairwise_distances(X)

    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    Deg = np.diag(1 / np.sum(G, axis=1)**0.5)
    K = Deg @ G @ Deg

    # Now K has the exact same eigenvalues as the diffusion matrix `P`
    # which is defined as `P = D^{-1} K`, with `D = np.diag(np.sum(K, axis=1))`.

    return K

def diffusion_spectral_entropy(embedding_vectors: np.array,
                               gaussian_kernel_sigma: float = 10,
                               t: int = 1,
                               max_N: int = 10000,
                               chebyshev_approx: bool = False,
                               eigval_save_path: str = None,
                               eigval_save_precision: np.dtype = np.float16,
                               classic_shannon_entropy: bool = False,
                               matrix_entry_entropy: bool = False,
                               num_bins_per_dim: int = 2,
                               random_seed: int = 0,
                               verbose: bool = False):
    '''
    >>> If `classic_shannon_entropy` is False (default)

    Diffusion Spectral Entropy over a set of N vectors, each of D dimensions.

    DSE = - sum_i [eig_i^t log eig_i^t]
        where each `eig_i` is an eigenvalue of `P`,
        where `P` is the diffusion matrix computed on the data graph of the [N, D] vectors.

    >>> If `classic_shannon_entropy` is True

    Classic Shannon Entropy over a set of N vectors, each of D dimensions.

    CSE = - sum_i [p(x) log p(x)]
        where each p(x) is the probability density of a histogram bin, after some sort of binning.

    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        gaussian_kernel_sigma: float
            The bandwidth of Gaussian kernel (for computation of the diffusion matrix)
            Can be adjusted per the dataset.
            Increase if the data points are very far away from each other.

        t: int
            Power of diffusion matrix (equivalent to power of diffusion eigenvalues)
            <-> Iteration of diffusion process
            Usually small, e.g., 1 or 2.
            Can be adjusted per dataset.
            Rule of thumb: after powering eigenvalues to `t`, there should be approximately
                           1 percent of eigenvalues that remain larger than 0.01

        max_N: int
            Max number of data points / samples used for computation.

        chebyshev_approx: bool
            Whether or not to use Chebyshev moments for faster approximation of eigenvalues.
            Currently we DO NOT RECOMMEND USING THIS. Eigenvalues may be changed quite a bit.

        eigval_save_path: str
            If provided,
                (1) If running for the first time, will save the computed eigenvalues in this location.
                (2) Otherwise, if the file already exists, skip eigenvalue computation and load from this file.

        eigval_save_precision: np.dtype
            We use `np.float16` by default to reduce storage space required.
            For best precision, use `np.float64` instead.

        classic_shannon_entropy: bool
            Toggle between DSE and CSE. False (default) == DSE.

        matrix_entry_entropy: bool
            An alternative formulation where, instead of computing the entropy on
            diffusion matrix eigenvalues, we compute the entropy on diffusion matrix entries.
            Only relevant to DSE.

        num_bins_per_dim: int
            Number of bins per feature dim.
            Only relevant to CSE (i.e., `classic_shannon_entropy` is True).

        verbose: bool
            Whether or not to print progress to console.
    '''

    # Subsample embedding vectors if number of data sample is too large.
    if max_N is not None and embedding_vectors is not None and len(
            embedding_vectors) > max_N:
        if random_seed is not None:
            random.seed(random_seed)
        rand_inds = np.array(
            random.sample(range(len(embedding_vectors)), k=max_N))
        embedding_vectors = embedding_vectors[rand_inds, :]

    if not classic_shannon_entropy:
        # Computing Diffusion Spectral Entropy.
        if verbose: print('Computing Diffusion Spectral Entropy...')

        if matrix_entry_entropy:
            if verbose: print('Computing diffusion matrix.')
            # Compute diffusion matrix `P`.
            K = compute_diffusion_matrix(embedding_vectors,
                                         sigma=gaussian_kernel_sigma)
            # Row normalize to get proper row stochastic matrix P
            D_inv = np.diag(1.0 / np.sum(K, axis=1))
            P = D_inv @ K

            if verbose: print('Diffusion matrix computed.')

            entries = P.reshape(-1)
            entries = np.abs(entries)
            prob = entries / entries.sum()

        else:
            if eigval_save_path is not None and os.path.exists(
                    eigval_save_path):
                if verbose:
                    print('Loading pre-computed eigenvalues from %s' %
                          eigval_save_path)
                eigvals = np.load(eigval_save_path)['eigvals']
                eigvals = eigvals.astype(
                    np.float64)  # mitigate rounding error.
                if verbose: print('Pre-computed eigenvalues loaded.')

            else:
                if verbose: print('Computing diffusion matrix.')
                # Note that `K` is a symmetric matrix with the same eigenvalues as the diffusion matrix `P`.
                K = compute_diffusion_matrix(embedding_vectors,
                                             sigma=gaussian_kernel_sigma)
                if verbose: print('Diffusion matrix computed.')

                if verbose: print('Computing eigenvalues.')
                if chebyshev_approx:
                    if verbose: print('Using Chebyshev approximation.')
                    eigvals = approx_eigvals(K)
                else:
                    eigvals = exact_eigvals(K)
                if verbose: print('Eigenvalues computed.')

                if eigval_save_path is not None:
                    os.makedirs(os.path.dirname(eigval_save_path),
                                exist_ok=True)
                    # Save eigenvalues.
                    eigvals = eigvals.astype(
                        eigval_save_precision)  # reduce storage space.
                    with open(eigval_save_path, 'wb+') as f:
                        np.savez(f, eigvals=eigvals)
                    if verbose:
                        print('Eigenvalues saved to %s' % eigval_save_path)

            # Eigenvalues may be negative. Only care about the magnitude, not the sign.
            eigvals = np.abs(eigvals)

            # Power eigenvalues to `t` to mitigate effect of noise.
            eigvals = eigvals**t

            prob = eigvals / eigvals.sum()

    else:
        # Computing Classic Shannon Entropy.
        if verbose: print('Computing Classic Shannon Entropy...')

        vecs = embedding_vectors.copy()

        # Min-Max scale each dimension.
        vecs = (vecs - np.min(vecs, axis=0)) / (np.max(vecs, axis=0) -
                                                np.min(vecs, axis=0))

        # Bin along each dimension.
        bins = np.linspace(0, 1, num_bins_per_dim + 1)[:-1]
        vecs = np.digitize(vecs, bins=bins)

        # Count probability.
        counts = np.unique(vecs, axis=0, return_counts=True)[1]
        prob = counts / np.sum(counts)

    prob = prob + np.finfo(float).eps
    entropy = -np.sum(prob * np.log2(prob))

    return entropy

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
) -> float:
    """
    Wrapper for diffusion_spectral_entropy.

    Parameters:
        embeddings: Input data array
        dataset: Provided for protocol compliance (unused)
        module: Provided for protocol compliance (unused)
        t: Diffusion time for entropy mode
        gaussian_kernel_sigma: Bandwidth of Gaussian kernel
        output_mode: "entropy" (default) or "eigenvalue_count"
        t_high: High diffusion time for asymptotic eigenvalue counting
        numerical_floor: Numerical noise floor for eigenvalue counting (not methodological)
        max_N: Max samples for eigenvalue computation (default 10000). Subsample if larger.
        random_seed: Seed for reproducible subsampling

    output_mode options:
        "entropy": Return DSE value using parameter t
        "eigenvalue_count": Return count of eigenvalues that persist at high diffusion time.
                           At high t, only eigenvalues for disconnected components persist
                           (don't decay to 0). This sidesteps threshold-picking by relying
                           on asymptotic behavior. The numerical_floor is just to filter
                           floating-point noise, not a methodological parameter.
        "eigenvalue_count_full": Same as eigenvalue_count but returns dict with spectrum
                                for debugging/logging (e.g., WandB histogram).
        "eigenvalue_count_sweep": Sweep across multiple t values efficiently.
                                 Computes eigenvalues once, then powers to each t.
                                 Returns dict with counts for each t value.
                                 Pass t_high as a list: [10, 50, 100, 200, 500]
    """
    # Subsample if too large (O(n²) memory for diffusion matrix)
    X = embeddings
    if max_N is not None and len(X) > max_N:
        random.seed(random_seed)
        rand_inds = np.array(random.sample(range(len(X)), k=max_N))
        X = X[rand_inds, :]

    if output_mode in ("eigenvalue_count", "eigenvalue_count_full", "eigenvalue_count_sweep"):
        # Compute diffusion matrix and eigenvalues (reuse existing functions)
        K = compute_diffusion_matrix(X, sigma=gaussian_kernel_sigma)
        eigvals = exact_eigvals(K)
        eigvals = np.abs(eigvals)

        if output_mode == "eigenvalue_count_sweep":
            # Sweep across multiple t values efficiently
            # Returns flat keys for WandB logging compatibility
            t_values = t_high if isinstance(t_high, (list, tuple)) else [t_high]
            results = {"raw_eigvals": eigvals}
            for t_val in t_values:
                eigvals_powered = eigvals ** t_val
                count = float(np.sum(eigvals_powered > numerical_floor))
                # Flat keys: count_t10, count_t50, spectrum_t10, spectrum_t50, ...
                results[f"count_t{t_val}"] = count
                results[f"spectrum_t{t_val}"] = eigvals_powered
            return results

        # Single t_high value
        eigvals_powered = eigvals ** t_high
        count = float(np.sum(eigvals_powered > numerical_floor))

        if output_mode == "eigenvalue_count_full":
            return {
                "count": count,
                "spectrum": eigvals_powered,
                "above_floor": eigvals_powered[eigvals_powered > numerical_floor],
            }
        return count

    # Default: return entropy (pass max_N to internal function for consistency)
    return diffusion_spectral_entropy(embeddings, t=t, gaussian_kernel_sigma=gaussian_kernel_sigma, max_N=max_N, random_seed=random_seed)