import functools

import numpy as np

from .latent_module_base import LatentModule, _to_numpy, _to_output


def _nnls(M, b):
    """Solve ``min ||M x - b||`` subject to ``x >= 0``; return ``x``.

    scipy 1.14.1 ``nnls`` stops at ``3 * M.shape[1]`` iterations and raises
    ``RuntimeError``. The archetypes beta step is wide (one unknown per sample,
    ``n_features + 1`` equations), and there it often reaches that cap. A wide
    system goes to the bounded-variable solver ``lsq_linear(method="bvls")``,
    which solves the same problem; any other system tries ``nnls`` first and
    falls back to it.
    """
    from scipy.optimize import lsq_linear, nnls

    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if M.shape[1] <= M.shape[0]:
        try:
            return nnls(M, b)[0]
        except RuntimeError:
            pass
    # bvls can return round-off below zero (about -1e-17); the bound is exact.
    return np.maximum(lsq_linear(M, b, bounds=(0, np.inf), method="bvls").x, 0.0)


# _optimize_alphas and _aa_simple are adapted from archetypes 0.6.2
# (archetypes/algorithms/archetypes.py), BSD 3-Clause License,
# Copyright (c) 2021, Aleix Alcacer Sales. The only change is the solver: _nnls in
# place of scipy.optimize.nnls.
def _optimize_alphas(B, A):
    B = np.pad(B, ((0, 0), (0, 1)), "constant", constant_values=200)
    A = np.pad(A, ((0, 0), (0, 1)), "constant", constant_values=200)
    alphas = np.empty((B.shape[0], A.shape[0]))
    for j in range(alphas.T.shape[1]):
        alphas.T[:, j] = _nnls(A.T, B.T[:, j])
    alphas /= alphas.sum(1)[:, None]
    alphas[np.isnan(alphas)] = 1 / alphas.shape[1]
    return alphas


def _aa_simple(X, i_alphas, i_betas, max_iter, tol):
    alphas, betas = i_alphas, i_betas
    Z = betas @ X
    rss_0 = np.inf
    for n_iter in range(max_iter):
        alphas = _optimize_alphas(X, Z)
        Z = np.linalg.pinv(alphas) @ X
        betas = _optimize_alphas(Z, X)
        Z = betas @ X
        rss = np.linalg.norm(X - alphas @ Z)  # Frobenius norm
        if np.abs(rss_0 - rss) < tol:
            break
        rss_0 = rss
    return alphas, betas, rss_0, Z, n_iter


@functools.cache
def _archetypes_aa_class():
    """Return ``_ArchetypesAA``, a private subclass of ``archetypes.AA`` (0.6 API).

    ``archetypes`` is imported here, not at module level, so that
    ``import manylatents.algorithms.latent`` does not import it.

    The subclass carries two shims and a solver:

    - ``_validate_data``: archetypes 0.6 calls ``BaseEstimator._validate_data``,
      which scikit-learn 1.6 removed. The shim calls ``check_array``.
    - ``_init_coefs``: when ``n_samples <= n_archetypes * n_features``,
      ``furthest_sum`` builds a distance matrix with ``np.mat``, which NumPy 2.0
      removed. The shim builds that matrix itself and passes it in.
    - ``fit`` and ``transform`` run in float64 and solve with :func:`_nnls`, not
      plain scipy ``nnls``, which stops at its iteration cap on float32 input and
      on many archetypes.
    """
    from archetypes import AA
    from archetypes.algorithms.furthest_sum import furthest_sum
    from sklearn.utils.validation import (
        check_array,
        check_is_fitted,
        check_random_state,
    )

    class _ArchetypesAA(AA):
        def _validate_data(self, X, **kwargs):
            return check_array(X, **kwargs)

        def _init_coefs(self, X, random_state):
            if (
                self._algorithm_init == "random"
                or X.shape[0] > self.n_archetypes * X.shape[1]
            ):
                return super()._init_coefs(X, random_state)

            # Same draws, in the same order, as AA._init_coefs in archetypes 0.6.2.
            ind = random_state.choice(self.n_archetypes, X.shape[0])
            alphas = np.zeros((X.shape[0], self.n_archetypes), dtype=np.float64)
            for i, j in enumerate(ind):
                alphas[i, j] = 1

            # The distance matrix of furthest_sum's kernel branch, without np.mat.
            # Exact symmetry makes furthest_sum use it as given.
            G = X @ X.T
            g = np.diag(G)
            D = np.lib.scimath.sqrt(g[None, :] - 2 * G + g[:, None])
            D = (D + D.T) / 2

            betas = np.zeros((self.n_archetypes, X.shape[0]), dtype=np.float64)
            for i, j in enumerate(furthest_sum(D, self.n_archetypes, random_state)):
                betas[i, j] = 1
            return alphas, betas

        def fit(self, X, y=None, **fit_params):
            # AA.fit with float64 input and _aa_simple; n_init, rss_ and the
            # fitted attributes are as in archetypes 0.6.2.
            X = self._validate_data(X, dtype=np.float64)
            self._check_parameters()
            self._check_data(X)
            random_state = check_random_state(self.random_state)
            self.rss_ = np.inf
            for _ in range(self.n_init):
                i_alphas, i_betas = self._init_coefs(X, random_state)
                alphas, betas, rss, Z, n_iter = _aa_simple(
                    X, i_alphas, i_betas, self.max_iter, self.tol
                )
                if rss < self.rss_:
                    self.alphas_, self.betas_, self.archetypes_ = alphas, betas, Z
                    self.n_iter_, self.rss_ = n_iter, rss
            return self

        def transform(self, X):
            check_is_fitted(self)
            X = self._validate_data(X, dtype=np.float64)
            self._check_parameters()
            return _optimize_alphas(X, self.archetypes_)

    return _ArchetypesAA


class ArchetypalAnalysisModule(LatentModule):
    def __init__(
        self,
        n_components: int = 3,
        method: str = "pgd",
        max_iter: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        # ``method`` is kept so that existing configs (``method: pgd``) still
        # instantiate. The archetypes 0.6 API has no ``method``, so it is not
        # forwarded. ``max_iter`` is the number of outer iterations of AA.
        super().__init__(n_components=n_components, 
                         init_seed=random_state,
                         **kwargs)
        self.method = method
        self.max_iter = max_iter
        self.model = None
        self._is_fitted = False

    def fit(self, x, y=None) -> None:
        """Fits the archetypal analysis model to the input data."""
        x_np = _to_numpy(x)
        self.model = _archetypes_aa_class()(
            n_archetypes=self.n_components,
            max_iter=self.max_iter,
            random_state=self.init_seed,
        )
        self.model.fit(x_np)
        self._is_fitted = True

    def transform(self, x):
        """Transforms the input into similarity degrees to each archetype."""
        if not self._is_fitted:
            raise RuntimeError("Archetypal Analysis model is not fitted yet. Call `fit` first.")

        x_np = _to_numpy(x)
        similarity = self.model.transform(x_np)
        return _to_output(similarity, x)
