"""Isomap: classical MDS on graph geodesic distances (sklearn backend)."""
from typing import Optional

from sklearn.manifold import Isomap

from .latent_module_base import LatentModule, _to_numpy, _to_output
from ...utils.kwargs import route_kwargs


class IsomapModule(LatentModule):
    """Isomap via ``sklearn.manifold.Isomap``.

    Parameters
    ----------
    n_components : int
        Number of output coordinates.
    random_state : int | None
        Seed stored as ``init_seed``. ``sklearn.manifold.Isomap`` takes no seed and its
        result does not depend on one, so this value does not reach sklearn.
    n_neighbors : int | None
        Size of the kNN graph that defines the geodesic distances. Overridden by
        ``neighborhood_size`` when that is given. The module default is 10; the
        ``sklearn.manifold.Isomap`` default is 5. When ``radius`` is given and
        ``neighborhood_size`` is not, the module sets ``n_neighbors`` to ``None`` so that
        sklearn builds a radius graph.
    eigen_solver, tol, max_iter, path_method, neighbors_algorithm, metric, n_jobs
        Passed to ``sklearn.manifold.Isomap`` unchanged.
    fit_fraction : float
        Fraction of the input rows used for fitting (the first rows).
    neighborhood_size : int | None
        The shared neighborhood parameter. Maps to sklearn ``n_neighbors``.
    **kwargs
        Other ``sklearn.manifold.Isomap`` parameters (for example ``p`` or ``radius``).
        A key that ``Isomap`` does not accept raises ``TypeError``.
    """

    _supports_input_metric = True

    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: Optional[int] = 10,
        eigen_solver: str = "auto",
        tol: float = 0,
        max_iter: Optional[int] = None,
        path_method: str = "auto",
        neighbors_algorithm: str = "auto",
        metric: str = "minkowski",
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1.0,
        neighborhood_size: Optional[int] = None,
        input_metric: Optional[str] = None,
        **kwargs,
    ):
        datamodule = kwargs.pop("datamodule", None)
        extra = route_kwargs(Isomap, kwargs, context="IsomapModule")
        super().__init__(
            n_components=n_components, init_seed=random_state,
            neighborhood_size=neighborhood_size, input_metric=input_metric, datamodule=datamodule,
        )
        if input_metric is not None:
            metric = input_metric
        self.random_state = random_state
        if neighborhood_size is not None:
            n_neighbors = neighborhood_size
        elif extra.get("radius") is not None:
            n_neighbors = None
        self.n_neighbors = n_neighbors
        self.fit_fraction = fit_fraction
        self.model = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=n_components,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            path_method=path_method,
            neighbors_algorithm=neighbors_algorithm,
            metric=metric,
            n_jobs=n_jobs,
            **extra,
        )

    def fit(self, x, y=None) -> None:
        """Fit Isomap on the first ``fit_fraction`` of the rows."""
        x_np = _to_numpy(x)
        n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x):
        """Embed rows with the fitted model (sklearn out-of-sample extension)."""
        if not self._is_fitted:
            raise RuntimeError("Isomap model is not fitted yet. Call `fit` first.")
        return _to_output(self.model.transform(_to_numpy(x)), x)
