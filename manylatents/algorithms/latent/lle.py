"""Locally linear embedding and its variants (sklearn backend)."""
from typing import Optional

from sklearn.manifold import LocallyLinearEmbedding

from .latent_module_base import LatentModule, _to_numpy, _to_output
from ...utils.kwargs import route_kwargs

_VALID_METHODS = {"standard", "hessian", "modified", "ltsa"}


class LocallyLinearModule(LatentModule):
    """Locally linear embedding via ``sklearn.manifold.LocallyLinearEmbedding``.

    One module covers the four sklearn variants through ``method``:

    | method | algorithm |
    |---|---|
    | ``standard`` | LLE (Roweis and Saul 2000) |
    | ``hessian`` | Hessian eigenmaps (Donoho and Grimes 2003) |
    | ``modified`` | Modified LLE (Zhang and Wang 2007) |
    | ``ltsa`` | Local tangent space alignment (Zhang and Zha 2004) |

    Parameters
    ----------
    n_components : int
        Number of output coordinates.
    random_state : int | None
        Seed for the ``arpack`` eigen solver. Passed to sklearn ``random_state``.
    method : str
        One of ``standard``, ``hessian``, ``modified``, ``ltsa``.
    n_neighbors : int
        Size of each local neighborhood. Overridden by ``neighborhood_size`` when that is
        given. The ``hessian`` method needs ``n_neighbors > n_components * (n_components + 3) / 2``.
    reg, eigen_solver, tol, max_iter, hessian_tol, modified_tol, neighbors_algorithm, n_jobs
        Passed to ``sklearn.manifold.LocallyLinearEmbedding`` unchanged.
    fit_fraction : float
        Fraction of the input rows used for fitting (the first rows).
    neighborhood_size : int | None
        The shared neighborhood parameter. Maps to sklearn ``n_neighbors``.
    **kwargs
        Other ``LocallyLinearEmbedding`` parameters. A key that sklearn does not accept
        raises ``TypeError``.
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        method: str = "standard",
        n_neighbors: int = 10,
        reg: float = 1e-3,
        eigen_solver: str = "auto",
        tol: float = 1e-6,
        max_iter: int = 100,
        hessian_tol: float = 1e-4,
        modified_tol: float = 1e-12,
        neighbors_algorithm: str = "auto",
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1.0,
        neighborhood_size: Optional[int] = None,
        **kwargs,
    ):
        datamodule = kwargs.pop("datamodule", None)
        extra = route_kwargs(LocallyLinearEmbedding, kwargs, context="LocallyLinearModule")
        super().__init__(
            n_components=n_components, init_seed=random_state,
            neighborhood_size=neighborhood_size, datamodule=datamodule,
        )
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {sorted(_VALID_METHODS)}."
            )
        self.method = method
        self.random_state = random_state
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.fit_fraction = fit_fraction
        self.model = LocallyLinearEmbedding(
            n_neighbors=self.n_neighbors,
            n_components=n_components,
            reg=reg,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            method=method,
            hessian_tol=hessian_tol,
            modified_tol=modified_tol,
            neighbors_algorithm=neighbors_algorithm,
            random_state=random_state,
            n_jobs=n_jobs,
            **extra,
        )

    def fit(self, x, y=None) -> None:
        """Fit the embedding on the first ``fit_fraction`` of the rows."""
        x_np = _to_numpy(x)
        n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x):
        """Embed rows with the fitted model (sklearn barycenter extension)."""
        if not self._is_fitted:
            raise RuntimeError("LLE model is not fitted yet. Call `fit` first.")
        return _to_output(self.model.transform(_to_numpy(x)), x)

    def fit_transform(self, x, y=None):
        """Fit and return sklearn's ``embedding_`` for the fitted rows.

        For the fitted rows this is the exact eigenvector solution. ``transform`` on the
        same rows would instead re-embed them through barycenter weights, which only
        approximates it.
        """
        self.fit(x, y)
        if self.fit_fraction >= 1.0:
            return _to_output(self.model.embedding_, x)
        return self.transform(x)
