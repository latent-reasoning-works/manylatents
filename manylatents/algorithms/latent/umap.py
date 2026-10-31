
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from manylatents.params import sources as _p
from .latent_module_base import LatentModule, _to_numpy, _to_output
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import resolve_backend, resolve_device, torchdr_knn_to_dense
from ...utils.kwargs import route_kwargs


class UMAPModule(LatentModule):
    _supports_input_metric = True

    # Bounds mirror umap-learn 0.5.11 UMAP._validate_parameters; backend mirrors resolve_backend.
    _N_NEIGHBORS_DOCS = {  # noqa: RUF012 - read-only
        "lo": 2, "hi": 100, "scale": "log",
        "source": {"kind": "docs",
                   "ref": "umap-learn 0.5.11 UMAP docstring, n_neighbors: "
                          "'In general values should be in the range 2 to 100.'"},
    }
    PARAMS = {  # noqa: RUF012 - a read-only declaration source
        "n_components": {"min": 1, "role": "knob", "meaning": "Number of output coordinates."},
        "random_state": _p.seed("Seed; an int also makes umap-learn run on one thread."),
        # umap-learn truncates n_neighbors >= n_samples to n_samples - 1 (a warning).
        "n_neighbors": _p.neighbours(
            "Size of the local neighbourhood of the fuzzy graph.",
            "More global structure; local detail blurs.", "More local detail; the graph can fragment.",
            min=2, sensible=_N_NEIGHBORS_DOCS, limits=[{"op": "lt", "fact": "n_samples", "offset": 0}]),
        # min_dist <= spread, and this module leaves spread at its default 1.0.
        "min_dist": {"min": 0, "max": 1.0, "role": "knob",
                     "meaning": "Smallest distance between embedded points.",
                     "more": "Points spread out; looser clusters.", "less": "Points pack tightly."},
        "metric": {"type": "choice", "choices": _p.UMAP_METRICS, "role": "knob",
                   "meaning": "Distance between input rows."},
        "n_epochs": _p.iterations("Optimisation epochs; None picks 200 or 500 by dataset size.",
                                  min=0, also=[None]),
        "learning_rate": {"min": 0, "role": "knob", "meaning": "Initial step size of the layout optimisation.",
                          "more": "Larger steps; can be unstable.", "less": "Smaller steps; slower."},
        "fit_fraction": _p.fit_fraction(),
        "negative_sample_rate": {"min": 0, "also": [None], "role": "knob",
                                 "cost": {"time": "up", "memory": "none", "note": "more negative samples per edge"},
                                 "meaning": "Negative samples per positive edge; None keeps umap-learn's 5.",
                                 "more": "Stronger repulsion.", "less": "Weaker repulsion."},
        "repulsion_strength": {"min": 0, "also": [None], "role": "knob",
                               "meaning": "Weight of negative samples; None keeps umap-learn's 1.0.",
                               "more": "Clusters spread.", "less": "Clusters compress."},
        "init": {"type": "choice", "choices": ["pca", "spectral", "random", "tswspectral"], "role": "knob",
                 "meaning": "How the starting embedding is built; an array of start points also works."},
        "densmap": {"role": "knob", "meaning": "Use densMAP, which also preserves local density."},
        "dens_lambda": {"min": 0, "role": "knob", "meaning": "densMAP weight of the density term."},
        "dens_frac": {"min": 0, "max": 1, "role": "knob", "meaning": "densMAP fraction of epochs with the density term."},
        "dens_var_shift": {"min": 0, "role": "knob", "meaning": "densMAP regulariser of the local radius variance."},
        "backend": _p.backend(["sklearn", "torchdr", "auto"],
                              "Implementation; None or 'sklearn' uses umap-learn."),
        "device": _p.device(),
        "neighborhood_size": _p.neighbours(
            "Shared neighbour count; when set it replaces n_neighbors.",
            "More global structure; local detail blurs.", "More local detail; the graph can fragment.",
            min=2, also=[None], sensible=_N_NEIGHBORS_DOCS,
            limits=[{"op": "lt", "fact": "n_samples", "offset": 0}]),
        "input_metric": {"type": "choice", "choices": _p.UMAP_METRICS, "also": [None], "role": "knob",
                         "meaning": "Shared input metric; when set it replaces metric."},
    }

    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = 'euclidean',
        n_epochs: Optional[int] = 200,
        learning_rate: float = 1.0,
        fit_fraction: float = 1.0,
        negative_sample_rate: int | None = None,
        repulsion_strength: float | None = None,
        init: str | np.ndarray = "spectral",
        densmap: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        backend: str | None = None,
        device: str | None = None,
        neighborhood_size: Optional[int] = None,
        input_metric: Optional[str] = None,
        **kwargs
    ):
        datamodule = kwargs.pop("datamodule", None)
        self._extra_kwargs = kwargs   # forwarded to the umap backend in _create_model (routed, not blindly)
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device,
            neighborhood_size=neighborhood_size, input_metric=input_metric, datamodule=datamodule,
        )
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.min_dist = min_dist
        self.metric = input_metric if input_metric is not None else metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fit_fraction = fit_fraction
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.init = init
        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.random_state = random_state

        self._resolved_backend = resolve_backend(backend)
        self.model = self._create_model()

    def _create_model(self):
        if self._resolved_backend == "torchdr":
            if self.densmap:
                raise ValueError(
                    "densmap=True is not supported by the 'torchdr' UMAP backend; "
                    "use the default umap-learn backend for densMAP."
                )
            if not (isinstance(self.init, str) and self.init == "spectral"):
                raise ValueError(
                    "init is not supported by the 'torchdr' UMAP backend; "
                    "use the default umap-learn backend."
                )
            if self.repulsion_strength is not None:
                raise ValueError(
                    "repulsion_strength is not supported by the 'torchdr' UMAP backend; "
                    "use the default umap-learn backend."
                )
            from torchdr import UMAP

            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                device=resolve_device(self.device),
                random_state=self.random_state,
                **route_kwargs(UMAP, self._extra_kwargs, context="UMAPModule[torchdr]"),
            )
        else:
            from umap import UMAP

            kwargs = {}
            if self.negative_sample_rate is not None:
                kwargs["negative_sample_rate"] = self.negative_sample_rate
            if self.repulsion_strength is not None:
                kwargs["repulsion_strength"] = self.repulsion_strength
            if self.densmap:
                kwargs.update(
                    densmap=True,
                    dens_lambda=self.dens_lambda,
                    dens_frac=self.dens_frac,
                    dens_var_shift=self.dens_var_shift,
                )
            kwargs.update(route_kwargs(UMAP, self._extra_kwargs, context="UMAPModule[umap-learn]"))
            return UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
                init=self.init,
                **kwargs,
            )

    def fit(self, x, y=None) -> None:
        """Fits UMAP on a subset of data."""
        x_np = _to_numpy(x)
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        if isinstance(self.init, np.ndarray) and self.init.shape[0] != n_fit:
            raise ValueError(
                f"init array has {self.init.shape[0]} rows but the fit uses {n_fit} rows "
                f"(fit_fraction={self.fit_fraction}); pass an init array with one row per fitted sample."
            )

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            self.model.fit(x_torch)
        else:
            self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x):
        """Transforms data using the fitted UMAP model."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        x_np = _to_numpy(x)

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding_np = self.model.transform(x_torch).detach().cpu().numpy()
        else:
            embedding_np = self.model.transform(x_np)
        return _to_output(embedding_np, x)

    def fit_transform(self, x, y=None):
        """Fit and then transform on same data."""
        x_np = _to_numpy(x)

        if self._resolved_backend == "torchdr":
            import torch as th
            n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding_np = self.model.fit_transform(x_torch).detach().cpu().numpy()
            self._is_fitted = True
        else:
            embedding_np = self.model.fit_transform(x_np)
            self._is_fitted = True
        return _to_output(embedding_np, x)

    def affinity(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """Returns UMAP affinity matrix.

        UMAP's graph represents fuzzy membership strengths. This method can return
        either a row-stochastic (asymmetric) or symmetric diffusion operator version.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.
            use_symmetric: If True, return symmetric diffusion operator with guaranteed
                positive eigenvalues. If False, return row-stochastic matrix. Default False.

        Returns:
            N x N affinity matrix (row-normalized if use_symmetric=False, symmetric if True).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            K = self.kernel(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend == "torchdr":
                A = torchdr_knn_to_dense(self.model).cpu().numpy()
            else:
                A = np.asarray(self.model.graph_.todense())

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return A / row_sums

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Returns UMAP kernel matrix.

        For UMAP, the fuzzy simplicial set serves as both the kernel
        and affinity matrix. The graph already has zero diagonal.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
                Note: UMAP graph already has zero diagonal by construction.

        Returns:
            N x N kernel matrix (fuzzy simplicial set).
        """
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if self._resolved_backend == "torchdr":
            K = torchdr_knn_to_dense(self.model).cpu().numpy()
        else:
            K = np.asarray(self.model.graph_.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
