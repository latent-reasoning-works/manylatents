"""
Sammon mapping — locally-weighted metric MDS.

Sammon mapping minimizes the Sammon stress:

    E = (1 / sum_{i<j} d*_ij) * sum_{i<j} (d*_ij - d_ij)^2 / d*_ij

where ``d*`` is the ambient pairwise distance and ``d`` is the embedding
pairwise distance. The per-pair weight ``1/d*_ij`` emphasizes preservation
of small (local) distances relative to plain metric MDS.

Only metric MDS with euclidean distances is supported here. The optimization
is driven by ``scipy.optimize.minimize(method="L-BFGS-B")``: it is a
batteries-included quasi-Newton solver with analytic gradients, making the
result deterministic given the initialization — the classic-MDS init below
fixes the random seed, so the whole pipeline is reproducible without having
to hand-roll a gradient-descent loop.

The affinity / kernel interface mirrors ``MDSModule``: Sammon preserves the
same ambient pairwise distance matrix, so the spectral diagnostic (the
double-centered Gram matrix) plugs in identically.
"""

from typing import Optional

import numpy as np
import scipy.optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from .latent_module_base import LatentModule, _to_numpy, _to_output


class SammonMapping:
    """Core (non-Lightning) Sammon mapping implementation."""

    def __init__(
        self,
        ndim: int = 2,
        seed: Optional[int] = 42,
        distance_metric: str = "euclidean",
        max_iter: int = 500,
        tol: float = 1e-6,
        eps: float = 1e-10,
        verbose: bool = False,
    ):
        if distance_metric != "euclidean":
            raise ValueError(
                "SammonMapping currently only supports distance_metric='euclidean'; "
                f"got {distance_metric!r}."
            )
        self.ndim = ndim
        self.seed = seed
        self.distance_metric = distance_metric
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.verbose = verbose

        self.embedding: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.stress_: Optional[float] = None

    def _classic_init(self, D: np.ndarray) -> np.ndarray:
        """Classic MDS initialization (double-centered D^2 → PCA)."""
        D2 = D ** 2
        D2 = D2 - D2.mean(axis=0)[None, :]
        D2 = D2 - D2.mean(axis=1)[:, None]
        pca = PCA(
            n_components=self.ndim, svd_solver="randomized", random_state=self.seed
        )
        return pca.fit_transform(D2)

    def _stress_and_grad(self, Y_flat: np.ndarray, D_star: np.ndarray,
                         inv_D_star: np.ndarray, c: float,
                         n: int) -> tuple[float, np.ndarray]:
        """Sammon stress and its gradient wrt the embedding coordinates.

        Parameters
        ----------
        Y_flat : (n*ndim,) flattened embedding.
        D_star : (n, n) ambient distances (with self.eps on diagonal).
        inv_D_star : (n, n) 1/D_star (diagonal already zeroed).
        c : normalization constant = sum_{i<j} D_star_ij.
        n : number of samples.

        Returns
        -------
        (E, grad_flat) — scalar stress and (n*ndim,) gradient.
        """
        Y = Y_flat.reshape(n, self.ndim)
        # Embedding pairwise distances (add eps on diagonal to keep division safe).
        D = squareform(pdist(Y, metric="euclidean"))
        D_safe = D + self.eps * np.eye(n)

        diff = D_star - D                        # (n, n)
        # Stress: sum_{i,j} (d* - d)^2 / d*, divided by 2c (because we double-count).
        E = (diff ** 2 * inv_D_star).sum() / (2.0 * c)

        # Gradient of E wrt Y_k:
        #   dE/dY_k = -(2/c) * sum_{j != k} [(d*_kj - d_kj) / (d*_kj * d_kj)] * (Y_k - Y_j)
        coeff = (diff * inv_D_star) / D_safe      # (n, n); diagonal is ~0
        np.fill_diagonal(coeff, 0.0)
        # Vectorised accumulation: grad_k = -(2/c) * sum_j coeff[k, j] * (Y_k - Y_j)
        #                                 = -(2/c) * (Y_k * sum_j coeff[k,j] - sum_j coeff[k,j] * Y_j)
        row_sum = coeff.sum(axis=1, keepdims=True)  # (n, 1)
        grad = -(2.0 / c) * (row_sum * Y - coeff @ Y)
        return float(E), grad.ravel()

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Fit Sammon mapping to X and return the embedding."""
        D = squareform(pdist(X, metric=self.distance_metric))
        self.distance_matrix = D

        # Safe-divide copy of D used only inside the objective.
        D_star = D.copy()
        # Keep diagonal at 0 for the sum, but use eps when inverting.
        inv_D_star = np.zeros_like(D_star)
        off_diag = ~np.eye(D_star.shape[0], dtype=bool)
        inv_D_star[off_diag] = 1.0 / (D_star[off_diag] + self.eps)

        # Normalization constant c = sum_{i<j} d*_ij.
        c = D_star[np.triu_indices_from(D_star, k=1)].sum()
        if c <= 0:
            # Degenerate: all points identical → return zeros.
            self.embedding = np.zeros((X.shape[0], self.ndim))
            self.stress_ = 0.0
            return self.embedding

        n = X.shape[0]
        Y0 = self._classic_init(D).ravel()

        result = scipy.optimize.minimize(
            fun=self._stress_and_grad,
            x0=Y0,
            args=(D_star, inv_D_star, c, n),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": self.max_iter,
                "gtol": self.tol,
                "disp": self.verbose,
            },
        )

        Y = result.x.reshape(n, self.ndim)
        self.embedding = Y
        self.stress_ = float(result.fun)
        return Y


class SammonModule(LatentModule):
    """LatentModule wrapper around Sammon mapping."""

    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        distance_metric: str = "euclidean",
        max_iter: int = 500,
        tol: float = 1e-6,
        eps: float = 1e-10,
        verbose: bool = False,
        neighborhood_size: Optional[int] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            init_seed=random_state,
            neighborhood_size=neighborhood_size,
            backend=backend,
            device=device,
            **kwargs,
        )
        self.model = SammonMapping(
            ndim=n_components,
            seed=random_state,
            distance_metric=distance_metric,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
        )

    def fit(self, x, y=None) -> None:
        x_np = _to_numpy(x)
        self.model.embed(x_np)
        self._is_fitted = True

    def transform(self, x):
        """Sammon cannot out-of-sample; return the fitted embedding."""
        if not self._is_fitted:
            raise RuntimeError("Sammon model is not fitted yet. Call `fit` first.")
        return _to_output(self.model.embedding, x)

    def fit_transform(self, x, y=None):
        x_np = _to_numpy(x)
        embedding_np = self.model.embed(x_np)
        self._is_fitted = True
        return _to_output(embedding_np, x)

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Same Gram matrix as MDS (Sammon only re-weights the objective)."""
        return self.affinity(ignore_diagonal=ignore_diagonal)

    def affinity(self, ignore_diagonal: bool = False,
                 use_symmetric: bool = False) -> np.ndarray:
        """Double-centered Gram matrix (normalized by n-1), as in MDSModule.

        Sammon does not change the ambient distances — only the stress
        weighting — so the spectral diagnostic is identical to MDS.
        """
        if not self._is_fitted:
            raise RuntimeError("Sammon model is not fitted yet. Call `fit` first.")
        if self.model.distance_matrix is None:
            raise RuntimeError("Distance matrix not available. This should not happen.")

        D_squared = self.model.distance_matrix ** 2
        n = D_squared.shape[0]
        row_means = D_squared.mean(axis=1, keepdims=True)
        col_means = D_squared.mean(axis=0, keepdims=True)
        grand_mean = D_squared.mean()

        G = -0.5 * (D_squared - row_means - col_means + grand_mean)
        G = G / (n - 1)

        if ignore_diagonal:
            G = G - np.diag(np.diag(G))
        return G
