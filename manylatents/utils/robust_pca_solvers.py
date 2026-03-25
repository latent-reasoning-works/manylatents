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
