import functools

import numpy as np

from .latent_module_base import LatentModule, _to_numpy, _to_output


@functools.cache
def _archetypes_aa_class():
    """Return ``_ArchetypesAA``, a private subclass of ``archetypes.AA`` (0.6 API).

    ``archetypes`` is imported here, not at module level, so that
    ``import manylatents.algorithms.latent`` does not import it.

    The subclass carries two shims:

    - ``_validate_data``: archetypes 0.6 calls ``BaseEstimator._validate_data``,
      which scikit-learn 1.6 removed. The shim calls ``check_array``.
    - ``_init_coefs``: when ``n_samples <= n_archetypes * n_features``,
      ``furthest_sum`` builds a distance matrix with ``np.mat``, which NumPy 2.0
      removed. The shim builds that matrix itself and passes it in.
    """
    from archetypes import AA
    from archetypes.algorithms.furthest_sum import furthest_sum
    from sklearn.utils.validation import check_array

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
