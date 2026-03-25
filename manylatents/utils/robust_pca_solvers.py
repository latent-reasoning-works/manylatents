"""Global Robust PCA solvers: ADMM and Inexact ALM (IALM).

Implements the Principal Component Pursuit (PCP) decomposition:
    D = L + S
where L is low-rank and S is sparse.

References:
    - Candes et al., "Robust Principal Component Analysis?", JACM 2011
    - Lin et al., "The Augmented Lagrange Multiplier Method for Exact
      Recovery of Corrupted Low-Rank Matrices", 2010
    - Zhou et al., "Stable Principal Component Pursuit", ISIT 2010
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class SVTResult(NamedTuple):
    """Result of singular value thresholding."""

    matrix: np.ndarray
    U: np.ndarray
    sigma: np.ndarray
    Vt: np.ndarray


class RobustPCAResult(NamedTuple):
    """Result of a Robust PCA decomposition."""

    L: np.ndarray
    S: np.ndarray
    rank: int
    n_iter: int
    convergence_history: dict
    svd_factors: tuple  # (U, sigma, Vt) of the final L


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _shrink(X: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft thresholding (shrinkage operator).

    Returns sign(X) * max(|X| - tau, 0).
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _svt(
    X: np.ndarray,
    tau: float,
    prev_rank: int | None = None,
    rank_buffer: int = 5,
    use_truncated: bool = True,
) -> SVTResult:
    """Singular value thresholding with adaptive truncated SVD.

    Computes the proximal operator for the nuclear norm:
        SVT_tau(X) = U * max(sigma - tau, 0) * Vt

    When *use_truncated* is True and the expected rank is small relative
    to the matrix dimensions, uses ``scipy.sparse.linalg.svds`` for
    efficiency.  Falls back to full SVD when needed.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    tau : float
        Threshold for singular values.
    prev_rank : int | None
        Rank from the previous iteration (used to set truncated SVD size).
    rank_buffer : int
        Extra singular values to compute beyond *prev_rank*.
    use_truncated : bool
        Whether to attempt truncated SVD.

    Returns
    -------
    SVTResult
        Named tuple with fields (matrix, U, sigma, Vt) where *sigma*
        contains only the post-threshold (nonzero) singular values.
    """
    m, n = X.shape
    min_dim = min(m, n)

    # Decide whether truncated SVD is viable
    if use_truncated and prev_rank is not None and prev_rank >= 0:
        k = min(prev_rank + rank_buffer, min_dim - 1)
        if k >= 1 and k < min_dim:
            try:
                # svds returns singular values in ascending order
                U_k, sigma_k, Vt_k = svds(X, k=k)
                # Sort descending
                idx = np.argsort(sigma_k)[::-1]
                U_k = U_k[:, idx]
                sigma_k = sigma_k[idx]
                Vt_k = Vt_k[idx, :]

                # If smallest computed singular value > tau, we may have
                # missed some — fall back to full SVD
                if sigma_k[-1] > tau:
                    return _svt_full(X, tau)

                # Threshold
                mask = sigma_k > tau
                sigma_thresh = sigma_k[mask] - tau
                U_thresh = U_k[:, mask]
                Vt_thresh = Vt_k[mask, :]

                # Reconstruct with explicit parentheses for efficiency:
                # (U * sigma) @ Vt  — avoids forming a large intermediate
                matrix = (U_thresh * sigma_thresh[np.newaxis, :]) @ Vt_thresh
                return SVTResult(matrix, U_thresh, sigma_thresh, Vt_thresh)
            except Exception:
                # svds can fail on some matrices; fall back
                pass

    return _svt_full(X, tau)


def _svt_full(X: np.ndarray, tau: float) -> SVTResult:
    """Full SVD based singular value thresholding (fallback)."""
    U_full, sigma_full, Vt_full = np.linalg.svd(X, full_matrices=False)
    mask = sigma_full > tau
    sigma_thresh = sigma_full[mask] - tau
    U_thresh = U_full[:, mask]
    Vt_thresh = Vt_full[mask, :]

    if len(sigma_thresh) == 0:
        matrix = np.zeros_like(X)
        return SVTResult(matrix, U_thresh, sigma_thresh, Vt_thresh)

    matrix = (U_thresh * sigma_thresh[np.newaxis, :]) @ Vt_thresh
    return SVTResult(matrix, U_thresh, sigma_thresh, Vt_thresh)


# ---------------------------------------------------------------------------
# IALM solver
# ---------------------------------------------------------------------------


def rpca_ialm(
    D: np.ndarray,
    lmbda: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
    delta: float | None = None,
    mu: float | None = None,
    mu_max: float = 1e7,
    rho: float = 1.5,
    use_truncated_svd: bool = True,
    verbose: bool = False,
) -> RobustPCAResult:
    """Inexact Augmented Lagrange Multiplier (IALM) solver for Robust PCA.

    Solves the Principal Component Pursuit problem:
        min ||L||_* + lambda * ||S||_1  s.t.  D = L + S

    When *delta* is set, solves the Stable PCP variant:
        min ||L||_* + lambda * ||S||_1  s.t.  ||D - L - S||_F <= delta

    Parameters
    ----------
    D : np.ndarray
        Observed data matrix (m x n).
    lmbda : float | None
        Regularization parameter. Default: 1 / sqrt(max(m, n)).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on relative Frobenius error.
    delta : float | None
        Noise bound for Stable PCP. None for exact PCP.
    mu : float | None
        Initial augmented Lagrangian penalty parameter.
    mu_max : float
        Maximum value for mu.
    rho : float
        Multiplicative factor for increasing mu each iteration.
    use_truncated_svd : bool
        Whether to use truncated SVD for efficiency.
    verbose : bool
        Print iteration info.

    Returns
    -------
    RobustPCAResult
        Decomposition result with L, S, rank, iteration count,
        convergence history, and SVD factors of L.
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    # Guard: zero matrix
    D_norm = np.linalg.norm(D, "fro")
    if D_norm == 0.0:
        return RobustPCAResult(
            L=np.zeros_like(D),
            S=np.zeros_like(D),
            rank=0,
            n_iter=0,
            convergence_history={"error": [], "rank": [], "sparsity": []},
            svd_factors=(np.empty((m, 0)), np.empty(0), np.empty((0, n))),
        )

    # Defaults
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(m, n))

    # Initial dual variable: Y_0 = D / J(D)
    spectral_norm = np.linalg.norm(D, ord=2)
    inf_norm = np.linalg.norm(D, ord=np.inf)
    J_D = max(spectral_norm, inf_norm / lmbda)
    Y = D / J_D

    if mu is None:
        mu = 1e-5

    S = np.zeros_like(D)
    prev_rank = 0

    convergence_history: dict[str, list] = {"error": [], "rank": [], "sparsity": []}

    for k in range(1, max_iter + 1):
        # L step: SVT on (D - S + Y/mu)
        svt_result = _svt(
            D - S + Y / mu,
            tau=1.0 / mu,
            prev_rank=prev_rank,
            use_truncated=use_truncated_svd,
        )
        L = svt_result.matrix
        current_rank = len(svt_result.sigma)
        prev_rank = current_rank

        # S step: shrinkage on (D - L + Y/mu)
        S = _shrink(D - L + Y / mu, lmbda / mu)

        # Residual
        residual_matrix = D - L - S
        residual = np.linalg.norm(residual_matrix, "fro")
        rel_error = residual / D_norm

        # Record history
        sparsity = np.count_nonzero(S) / S.size if S.size > 0 else 0.0
        convergence_history["error"].append(rel_error)
        convergence_history["rank"].append(current_rank)
        convergence_history["sparsity"].append(sparsity)

        if verbose:
            print(
                f"IALM iter {k}: rank={current_rank}, "
                f"rel_error={rel_error:.2e}, mu={mu:.2e}"
            )

        # Convergence check
        if delta is not None:
            # Stable PCP: converge when residual <= delta
            if residual <= delta:
                break
            # Dual projection for stable PCP
            Y = Y + mu * residual_matrix
            Y_norm = np.linalg.norm(Y, "fro")
            if Y_norm > 0:
                scale = delta / max(residual, delta)
                Y *= scale
        else:
            # Exact PCP
            if rel_error < tol:
                break
            # Dual update
            Y = Y + mu * residual_matrix

        # Increase penalty
        mu = min(rho * mu, mu_max)

    # Final SVD factors for L
    svd_factors = (svt_result.U, svt_result.sigma, svt_result.Vt)

    return RobustPCAResult(
        L=L,
        S=S,
        rank=current_rank,
        n_iter=k,
        convergence_history=convergence_history,
        svd_factors=svd_factors,
    )


# ---------------------------------------------------------------------------
# ADMM solver
# ---------------------------------------------------------------------------


def rpca_admm(
    D: np.ndarray,
    lmbda: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-7,
    delta: float | None = None,
    mu: float | None = None,
    use_truncated_svd: bool = True,
    verbose: bool = False,
) -> RobustPCAResult:
    """Fixed-mu Augmented Lagrange Multiplier (ADMM) solver for Robust PCA.

    Same objective as :func:`rpca_ialm` but keeps the penalty parameter
    *mu* fixed throughout, which can be simpler but typically requires
    more iterations.

    Parameters
    ----------
    D : np.ndarray
        Observed data matrix (m x n).
    lmbda : float | None
        Regularization parameter. Default: 1 / sqrt(max(m, n)).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on relative Frobenius error.
    delta : float | None
        Noise bound for Stable PCP.
    mu : float | None
        Fixed penalty parameter. Default: m*n / (4 * sum(|D|)).
    use_truncated_svd : bool
        Whether to use truncated SVD for efficiency.
    verbose : bool
        Print iteration info.

    Returns
    -------
    RobustPCAResult
        Decomposition result.
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    # Guard: zero matrix
    D_norm = np.linalg.norm(D, "fro")
    if D_norm == 0.0:
        return RobustPCAResult(
            L=np.zeros_like(D),
            S=np.zeros_like(D),
            rank=0,
            n_iter=0,
            convergence_history={"error": [], "rank": [], "sparsity": []},
            svd_factors=(np.empty((m, 0)), np.empty(0), np.empty((0, n))),
        )

    # Defaults
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(m, n))

    abs_sum = np.sum(np.abs(D))
    if mu is None:
        if abs_sum > 0:
            mu = m * n / (4.0 * abs_sum)
        else:
            mu = 1.0

    # Initialize
    Y = np.zeros_like(D)
    S = np.zeros_like(D)
    prev_rank = 0

    convergence_history: dict[str, list] = {"error": [], "rank": [], "sparsity": []}

    for k in range(1, max_iter + 1):
        # L step: SVT on (D - S + Y/mu)
        svt_result = _svt(
            D - S + Y / mu,
            tau=1.0 / mu,
            prev_rank=prev_rank,
            use_truncated=use_truncated_svd,
        )
        L = svt_result.matrix
        current_rank = len(svt_result.sigma)
        prev_rank = current_rank

        # S step: shrinkage on (D - L + Y/mu)
        S = _shrink(D - L + Y / mu, lmbda / mu)

        # Residual
        residual_matrix = D - L - S
        residual = np.linalg.norm(residual_matrix, "fro")
        rel_error = residual / D_norm

        # Record history
        sparsity = np.count_nonzero(S) / S.size if S.size > 0 else 0.0
        convergence_history["error"].append(rel_error)
        convergence_history["rank"].append(current_rank)
        convergence_history["sparsity"].append(sparsity)

        if verbose:
            print(
                f"ADMM iter {k}: rank={current_rank}, "
                f"rel_error={rel_error:.2e}, mu={mu:.2e}"
            )

        # Convergence check
        if delta is not None:
            if residual <= delta:
                break
            # Dual projection for stable PCP
            Y = Y + mu * residual_matrix
            Y_norm = np.linalg.norm(Y, "fro")
            if Y_norm > 0:
                scale = delta / max(residual, delta)
                Y *= scale
        else:
            if rel_error < tol:
                break
            Y = Y + mu * residual_matrix

        # mu stays fixed (ADMM — no rho update)

    # Final SVD factors for L
    svd_factors = (svt_result.U, svt_result.sigma, svt_result.Vt)

    return RobustPCAResult(
        L=L,
        S=S,
        rank=current_rank,
        n_iter=k,
        convergence_history=convergence_history,
        svd_factors=svd_factors,
    )


# ---------------------------------------------------------------------------
# Robust Local PCA
# ---------------------------------------------------------------------------


class RobustLocalPCAResult(NamedTuple):
    """Result of robust local PCA analysis."""

    local_bases: np.ndarray       # (n, max_dim, d)
    local_eigenvalues: np.ndarray  # (n, min(k,d))
    local_dims: np.ndarray        # (n,)
    local_variances: np.ndarray   # (n,)
    outlier_masks: np.ndarray     # (n, k) bool
    condition_numbers: np.ndarray  # (n,)
    support_sizes: np.ndarray    # (n,) int


def _estimate_local_dim(eigenvalues: np.ndarray) -> int:
    """Estimate local dimensionality via largest log-eigenvalue gap.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues in descending order.

    Returns
    -------
    int
        Estimated local intrinsic dimensionality (>= 1).
    """
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) <= 1:
        return 1
    log_evals = np.log(pos)
    gaps = log_evals[:-1] - log_evals[1:]
    return int(np.argmax(gaps) + 1)


# ---------------------------------------------------------------------------
# Local covariance estimation methods
# ---------------------------------------------------------------------------
# All return (eigenvalues, eigenvectors, outlier_mask, support_size).
# Eigenvalues are in descending order.


def _local_cov_none(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Standard empirical covariance (no robustness)."""
    k, d = points.shape
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(k - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Flip to descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    outlier_mask = np.zeros(k, dtype=bool)
    return eigenvalues, eigenvectors, outlier_mask, k


def _local_cov_trimmed(
    points: np.ndarray,
    trim_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Trimmed covariance: remove furthest points from coordinate-wise median."""
    k, d = points.shape
    n_trim = int(k * trim_fraction)
    if n_trim == 0:
        return _local_cov_none(points)

    # Coordinate-wise median centroid
    centroid = np.median(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    # Mark the furthest n_trim points as outliers
    outlier_mask = np.zeros(k, dtype=bool)
    outlier_idx = np.argsort(dists)[-n_trim:]
    outlier_mask[outlier_idx] = True

    inliers = points[~outlier_mask]
    support_size = inliers.shape[0]
    centered = inliers - inliers.mean(axis=0)
    cov = centered.T @ centered / max(support_size - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors, outlier_mask, support_size


def _local_cov_mcd(
    points: np.ndarray,
    support_fraction: float = 0.75,
    random_state: int = 42,
    trim_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Minimum Covariance Determinant estimator via sklearn.

    Falls back to trimmed covariance when k <= d (MCD requires more
    samples than features).
    """
    k, d = points.shape
    if k <= d:
        return _local_cov_trimmed(points, trim_fraction=trim_fraction)

    from sklearn.covariance import MinCovDet

    try:
        mcd = MinCovDet(
            support_fraction=support_fraction,
            random_state=random_state,
        )
        mcd.fit(points)
        cov = mcd.covariance_
        support_mask = mcd.support_
        outlier_mask = ~support_mask
        support_size = int(support_mask.sum())
    except Exception:
        # MCD can fail on degenerate data; fall back
        return _local_cov_trimmed(points, trim_fraction=trim_fraction)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors, outlier_mask, support_size


def _local_cov_huber(
    points: np.ndarray,
    trim_fraction: float = 0.1,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Huber-style iterative reweighting for robust covariance.

    Falls back to trimmed covariance when k <= d.
    """
    from scipy.stats import chi2

    k, d = points.shape
    if k <= d:
        return _local_cov_trimmed(points, trim_fraction=trim_fraction)

    c = np.sqrt(chi2.ppf(0.95, df=d))

    # Initialize with empirical mean and covariance
    mu = points.mean(axis=0)
    centered = points - mu
    cov = centered.T @ centered / max(k - 1, 1)

    for _ in range(max_iter):
        # Regularize for inversion
        cov_reg = cov + 1e-10 * np.eye(d)
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            break

        # Mahalanobis distances
        diff = points - mu
        mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # Huber weights
        weights = np.where(mahal <= c, 1.0, c / np.maximum(mahal, 1e-10))

        # Weighted mean and covariance
        w_sum = weights.sum()
        if w_sum < 1e-10:
            break
        mu_new = (weights[:, None] * points).sum(axis=0) / w_sum
        diff_new = points - mu_new
        cov_new = (weights[:, None] * diff_new).T @ diff_new / w_sum

        # Check convergence
        if np.linalg.norm(mu_new - mu) < tol:
            mu = mu_new
            cov = cov_new
            break
        mu = mu_new
        cov = cov_new

    # Outlier mask: points with weight < 1 (i.e. mahal > c)
    cov_reg = cov + 1e-10 * np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov_reg)
        diff = points - mu
        mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        outlier_mask = mahal > c
    except np.linalg.LinAlgError:
        outlier_mask = np.zeros(k, dtype=bool)

    support_size = int((~outlier_mask).sum())

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors, outlier_mask, support_size


def _local_cov_gaussian(
    points: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Gaussian kernel-weighted covariance.

    Adaptive bandwidth: epsilon = median(distances^2).
    Weights = exp(-dist^2 / epsilon).
    Outlier flag: weight < 0.1/k.
    """
    k, d = points.shape
    dist_sq = distances ** 2
    epsilon = np.median(dist_sq)
    if epsilon < 1e-20:
        epsilon = 1e-20

    weights = np.exp(-dist_sq / epsilon)

    # Weighted mean
    w_sum = weights.sum()
    if w_sum < 1e-10:
        return _local_cov_none(points)

    mu = (weights[:, None] * points).sum(axis=0) / w_sum
    diff = points - mu
    cov = (weights[:, None] * diff).T @ diff / w_sum

    # Outlier mask
    outlier_mask = weights < (0.1 / k)
    support_size = int((~outlier_mask).sum())

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors, outlier_mask, support_size


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def robust_local_pca(
    X: np.ndarray,
    n_neighbors: int = 20,
    n_components: int | None = None,
    robust_method: str = "trimmed",
    support_fraction: float = 0.75,
    trim_fraction: float = 0.1,
    precomputed_neighbors: np.ndarray | None = None,
    precomputed_distances: np.ndarray | None = None,
    cache: dict | None = None,
    random_state: int = 42,
) -> RobustLocalPCAResult:
    """Robust local PCA analysis for each point's neighborhood.

    For each point, computes a robust local covariance in its k-nearest
    neighborhood, then extracts local principal directions and estimates
    local intrinsic dimensionality.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n, d).
    n_neighbors : int
        Number of neighbors for each point.
    n_components : int | None
        Fixed number of local dimensions. If None, estimated automatically
        via log-eigenvalue gap.
    robust_method : str
        One of ``'none'``, ``'trimmed'``, ``'mcd'``, ``'huber'``,
        ``'gaussian'``.
    support_fraction : float
        Support fraction for MCD estimator.
    trim_fraction : float
        Fraction of points to trim in trimmed/fallback methods.
    precomputed_neighbors : np.ndarray | None
        Precomputed neighbor indices (n, k). If None, computed internally.
    precomputed_distances : np.ndarray | None
        Precomputed neighbor distances (n, k).
    cache : dict | None
        Optional kNN cache dict.
    random_state : int
        Random state for MCD estimator.

    Returns
    -------
    RobustLocalPCAResult
        Named tuple with local bases, eigenvalues, dimensions, variances,
        outlier masks, condition numbers, and support sizes.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Get neighborhoods
    if precomputed_neighbors is not None:
        indices = precomputed_neighbors
        distances = precomputed_distances  # may be None
        k = indices.shape[1]
    else:
        from manylatents.utils.knn import compute_knn

        distances, indices = compute_knn(
            X, k=n_neighbors, include_self=False, cache=cache,
        )
        k = n_neighbors

    max_eig_dim = min(k, d)

    # Storage
    all_eigenvalues = np.zeros((n, max_eig_dim), dtype=np.float64)
    all_eigenvectors = []  # list of (d, d) or (d, max_eig_dim) arrays
    all_dims = np.zeros(n, dtype=np.int64)
    all_variances = np.zeros(n, dtype=np.float64)
    all_outlier_masks = np.zeros((n, k), dtype=bool)
    all_condition_numbers = np.zeros(n, dtype=np.float64)
    all_support_sizes = np.zeros(n, dtype=np.int64)

    for i in range(n):
        nbr_points = X[indices[i]]

        # Dispatch to the appropriate method
        if robust_method == "none":
            evals, evecs, outlier_mask, support_size = _local_cov_none(
                nbr_points,
            )
        elif robust_method == "trimmed":
            evals, evecs, outlier_mask, support_size = _local_cov_trimmed(
                nbr_points, trim_fraction=trim_fraction,
            )
        elif robust_method == "mcd":
            evals, evecs, outlier_mask, support_size = _local_cov_mcd(
                nbr_points,
                support_fraction=support_fraction,
                random_state=random_state,
                trim_fraction=trim_fraction,
            )
        elif robust_method == "huber":
            evals, evecs, outlier_mask, support_size = _local_cov_huber(
                nbr_points, trim_fraction=trim_fraction,
            )
        elif robust_method == "gaussian":
            if distances is not None:
                dists_i = np.asarray(distances[i], dtype=np.float64)
            else:
                dists_i = np.linalg.norm(nbr_points - X[i], axis=1)
            evals, evecs, outlier_mask, support_size = _local_cov_gaussian(
                nbr_points, dists_i,
            )
        else:
            raise ValueError(
                f"Unknown robust_method: {robust_method!r}. "
                f"Must be one of 'none', 'trimmed', 'mcd', 'huber', 'gaussian'."
            )

        # Clamp eigenvalues to >= 0
        evals = np.maximum(evals, 0.0)

        # Estimate local dimension
        if n_components is None:
            dim = _estimate_local_dim(evals)
        else:
            dim = n_components

        # Store results
        n_store = min(len(evals), max_eig_dim)
        all_eigenvalues[i, :n_store] = evals[:n_store]
        all_eigenvectors.append(evecs)
        all_dims[i] = dim
        all_variances[i] = evals[:dim].sum() if dim <= len(evals) else evals.sum()
        all_outlier_masks[i, :len(outlier_mask)] = outlier_mask
        all_support_sizes[i] = support_size

        # Condition number
        dim_idx = min(dim, len(evals)) - 1
        if evals[0] > 1e-10 and dim > 0 and dim_idx >= 0 and evals[dim_idx] > 1e-10:
            all_condition_numbers[i] = evals[0] / evals[dim_idx]
        else:
            all_condition_numbers[i] = np.inf

    # Build local_bases: (n, max_dim, d)
    if n_components is not None:
        max_dim = n_components
    else:
        max_dim = int(all_dims.max()) if all_dims.max() > 0 else 1

    local_bases = np.zeros((n, max_dim, d), dtype=np.float64)
    for i in range(n):
        dim = int(all_dims[i])
        evecs = all_eigenvectors[i]
        # eigenvectors from eigh are columns; take top `dim` and transpose
        # evecs[:, :dim] has shape (d, dim); transposed = (dim, d)
        actual_dim = min(dim, evecs.shape[1])
        local_bases[i, :actual_dim, :] = evecs[:, :actual_dim].T

    return RobustLocalPCAResult(
        local_bases=local_bases,
        local_eigenvalues=all_eigenvalues,
        local_dims=all_dims,
        local_variances=all_variances,
        outlier_masks=all_outlier_masks,
        condition_numbers=all_condition_numbers,
        support_sizes=all_support_sizes,
    )
