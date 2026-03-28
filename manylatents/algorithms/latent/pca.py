
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output

_VALID_METHODS = {'standard', 'robust_admm', 'robust_ialm', 'robust_local'}


class PCAModule(LatentModule):
    """PCA with optional Robust PCA (RPCA) via ADMM or IALM solvers.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.
    random_state : int
        Random seed for reproducibility.
    fit_fraction : float
        Fraction of input samples used for fitting (subsampling).
    method : str
        Decomposition method: ``'standard'`` (sklearn PCA),
        ``'robust_ialm'`` (inexact ALM), ``'robust_admm'`` (ADMM),
        or ``'robust_local'`` (placeholder, not yet implemented).
    lmbda : float | None
        RPCA regularization parameter.  Default ``1/sqrt(max(m,n))``.
    solver_max_iter : int
        Maximum solver iterations for RPCA.
    tol : float
        Convergence tolerance for RPCA.
    delta : float | None
        Noise bound for Stable PCP variant.
    mu : float | None
        Initial augmented Lagrangian penalty.
    mu_max : float
        Maximum penalty (IALM only).
    rho : float
        Penalty growth factor (IALM only).
    use_truncated_svd : bool
        Use truncated SVD inside the solver for speed.
    n_neighbors : int
        Number of neighbors for ``robust_local`` (placeholder).
    robust_method : str
        Covariance method for ``robust_local`` (placeholder).
    support_fraction : float
        Support fraction for ``robust_local`` (placeholder).
    trim_fraction : float
        Trim fraction for ``robust_local`` (placeholder).
    verbose : bool
        Print solver progress.
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 42,
        fit_fraction: float = 1.0,
        # --- method dispatch ---
        method: str = 'standard',
        # --- global RPCA params ---
        lmbda: float | None = None,
        solver_max_iter: int = 500,
        tol: float = 1e-7,
        delta: float | None = None,
        mu: float | None = None,
        mu_max: float = 1e7,
        rho: float = 1.5,
        use_truncated_svd: bool = True,
        # --- local RPCA params (placeholder) ---
        n_neighbors: int = 20,
        robust_method: str = 'trimmed',
        support_fraction: float = 0.75,
        trim_fraction: float = 0.1,
        # --- misc ---
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            init_seed=random_state,
            **kwargs,
        )

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {sorted(_VALID_METHODS)}."
            )

        self.method = method
        self.fit_fraction = fit_fraction
        self.random_state = random_state
        self.verbose = verbose

        # Global RPCA params
        self.lmbda = lmbda
        self.solver_max_iter = solver_max_iter
        self.tol = tol
        self.delta = delta
        self.mu = mu
        self.mu_max = mu_max
        self.rho = rho
        self.use_truncated_svd = use_truncated_svd

        # Local RPCA params (placeholder)
        self.n_neighbors = (
            self.neighborhood_size
            if self.neighborhood_size is not None
            else n_neighbors
        )
        self.robust_method = robust_method
        self.support_fraction = support_fraction
        self.trim_fraction = trim_fraction

        # Only create sklearn PCA model for standard method
        self.model = None
        if self.method == 'standard':
            self.model = PCA(
                n_components=n_components,
                random_state=random_state,
            )

        self._is_fitted = False
        self._fit_data = None
        # Robust-specific state
        self._robust_result = None
        self._components = None
        self._mean = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, x, y=None) -> None:
        """Fit the model on data (or a fraction thereof)."""
        x_np = _to_numpy(x)
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        x_fit = x_np[:n_fit]

        if self.method == 'standard':
            self._fit_standard(x_fit)
        elif self.method in ('robust_ialm', 'robust_admm'):
            self._fit_robust_global(x_fit)
        elif self.method == 'robust_local':
            self._fit_robust_local(x_fit)
        self._is_fitted = True

    def _fit_standard(self, x_np: np.ndarray) -> None:
        """Standard sklearn PCA fit."""
        self.model.fit(x_np)
        self._fit_data = x_np

    def _fit_robust_global(self, x_np: np.ndarray) -> None:
        """Fit via global Robust PCA (ADMM or IALM)."""
        from manylatents.utils.robust_pca_solvers import rpca_admm, rpca_ialm

        solver = rpca_ialm if self.method == 'robust_ialm' else rpca_admm

        # Build solver kwargs — IALM has extra params (mu_max, rho)
        solver_kwargs = dict(
            lmbda=self.lmbda,
            max_iter=self.solver_max_iter,
            tol=self.tol,
            delta=self.delta,
            mu=self.mu,
            use_truncated_svd=self.use_truncated_svd,
            verbose=self.verbose,
        )
        if self.method == 'robust_ialm':
            solver_kwargs['mu_max'] = self.mu_max
            solver_kwargs['rho'] = self.rho

        result = solver(x_np, **solver_kwargs)
        self._robust_result = result

        # Extract principal components from the SVD of L
        U, sigma, Vt = result.svd_factors
        n_comp = min(self.n_components, len(sigma))
        self._components = Vt[:n_comp]
        self._mean = result.L.mean(axis=0)
        self._fit_data = result.L

    def _fit_robust_local(self, x_np):
        """Fit via robust local PCA with LTSA alignment."""
        from manylatents.utils.robust_pca_solvers import robust_local_pca, ltsa_align
        from manylatents.utils.knn import compute_knn

        self._X_fit_shape = tuple(x_np.shape)

        # Compute neighborhoods
        dists, indices = compute_knn(x_np.astype(np.float32),
                                      k=self.n_neighbors, include_self=False)

        # Robust local PCA
        local_result = robust_local_pca(
            x_np, n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            robust_method=self.robust_method,
            support_fraction=self.support_fraction,
            trim_fraction=self.trim_fraction,
            precomputed_neighbors=indices,
            precomputed_distances=dists,
            random_state=self.init_seed,
        )
        self._local_result = local_result

        # LTSA alignment
        self._embedding = ltsa_align(
            x_np.astype(np.float64), indices,
            local_result.local_bases, self.n_components
        ).astype(x_np.dtype)

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, x):
        """Transform data into the latent space."""
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")

        x_np = _to_numpy(x)

        if self.method == 'standard':
            embedding = self.model.transform(x_np)
        elif self.method in ('robust_ialm', 'robust_admm'):
            embedding = (x_np - self._mean) @ self._components.T
        elif self.method == 'robust_local':
            if hasattr(self, '_X_fit_shape') and tuple(x_np.shape) == self._X_fit_shape:
                embedding = self._embedding
            else:
                raise NotImplementedError(
                    "robust_local is transductive — transform() only supports "
                    "the original training data. Use fit_transform() instead."
                )

        return _to_output(embedding, x)

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(self, x, y=None):
        """Fit and transform in one call."""
        self.fit(x, y)
        if self.method == 'robust_local':
            # robust_local would cache embeddings during fit
            return _to_output(self._embedding, x)
        return self.transform(x)

    # ------------------------------------------------------------------
    # extra_outputs
    # ------------------------------------------------------------------

    def extra_outputs(self) -> dict:
        """Return extra outputs including robust decomposition artifacts."""
        extras = super().extra_outputs()

        if self.method in ('robust_ialm', 'robust_admm') and self._robust_result is not None:
            extras['low_rank_matrix'] = self._robust_result.L
            extras['sparse_matrix'] = self._robust_result.S
            extras['robust_rank'] = self._robust_result.rank
            extras['convergence_history'] = self._robust_result.convergence_history

        if self.method == 'robust_local' and hasattr(self, '_local_result'):
            r = self._local_result
            extras['local_eigenvalues'] = r.local_eigenvalues
            extras['local_dims'] = r.local_dims
            extras['local_variances'] = r.local_variances
            extras['outlier_masks'] = r.outlier_masks
            extras['condition_numbers'] = r.condition_numbers
            extras['support_sizes'] = r.support_sizes

        return extras

    # ------------------------------------------------------------------
    # kernel / affinity
    # ------------------------------------------------------------------

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Return the sample Gram matrix K = X_centered @ X_centered.T.

        For standard PCA, centres with ``self.model.mean_``.
        For global robust PCA, centres with the mean of the low-rank matrix L.

        Args:
            ignore_diagonal: If True, zero out the diagonal.

        Returns:
            N x N Gram matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")

        if self.method == 'robust_local':
            raise NotImplementedError("kernel not available for robust_local")

        if self.method == 'standard':
            mean = self.model.mean_
        else:
            mean = self._mean

        X_centered = self._fit_data - mean
        K = X_centered @ X_centered.T

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))

        return K

    def affinity(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """Return the normalised covariance matrix K / (n-1).

        Args:
            ignore_diagonal: If True, zero out the diagonal.
            use_symmetric: Ignored for PCA (always symmetric).

        Returns:
            N x N normalised covariance matrix.
        """
        K = self.kernel(ignore_diagonal=ignore_diagonal)
        n = self._fit_data.shape[0]
        return K / (n - 1)
