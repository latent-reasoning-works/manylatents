"""Leiden community detection as a LatentModule."""
import numpy as np
import scipy.sparse
import torch
from torch import Tensor

from manylatents.params import sources as _p

from .latent_module_base import LatentModule, _to_numpy, _to_output

_IMPLEMENTATIONS = ("leidenalg", "sknetwork")


def _libc():
    """The C library whose `rand()` scikit-network's compiled Leiden calls."""
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("c")
    return ctypes.CDLL(name) if name is not None else ctypes.cdll.msvcrt


class LeidenModule(LatentModule):
    """Cluster data using the Leiden algorithm on a kNN graph.

    Outputs cluster labels as a (N, 1) embedding. Uses the shared
    compute_knn() infrastructure for FAISS-GPU/CPU acceleration and caching.

    Parameters
    ----------
    resolution : float
        Leiden resolution parameter. Higher values produce more clusters.
    n_neighbors : int
        Number of neighbors for kNN graph construction.
    random_state : int
        Random seed for reproducibility.
    implementation : str
        ``"leidenalg"`` (default; leidenalg and python-igraph, GPL) or ``"sknetwork"``
        (scikit-network, BSD). The two use different resolution scales.
    """

    PARAMS = {  # noqa: RUF012 - a read-only declaration source
        "n_components": _p.unused("Unused; the output is one column of cluster labels."),
        # leidenalg 0.11.0 RBConfigurationVertexPartition checks no bound on the resolution.
        "resolution": {"role": "knob", "meaning": "Resolution of the RB configuration partition.",
                       "more": "More, smaller clusters.", "less": "Fewer, larger clusters."},
        # manylatents.utils.knn.compute_knn: 0 < k < n_samples.
        "n_neighbors": _p.neighbours(
            "Nearest neighbours of each point in the kNN graph.",
            "Denser graph; clusters merge.", "Sparser graph; clusters split.",
            limits=[{"op": "lt", "fact": "n_samples", "offset": 0}]),
        "random_state": _p.seed("Seed of the Leiden optimiser.", numpy=False),
        "neighborhood_size": _p.neighbours(
            "Shared neighbour count; when set it replaces n_neighbors.",
            "Denser graph; clusters merge.", "Sparser graph; clusters split.",
            also=[None], limits=[{"op": "lt", "fact": "n_samples", "offset": 0}]),
        "backend": _p.unused("Stored by the base class; Leiden does not use it.", "str") | {"also": [None]},
        "device": _p.unused("Stored by the base class; Leiden does not use it.", "str") | {"also": [None]},
        "implementation": {"type": "choice", "choices": list(_IMPLEMENTATIONS), "role": "knob",  # __init__ check
                           "meaning": "leidenalg (GPL) or sknetwork (BSD); different resolution scales, "
                                      "so one resolution can give other cluster counts."},
    }

    def __init__(
        self,
        n_components: int = 1,
        resolution: float = 0.5,
        n_neighbors: int = 15,
        random_state: int = 42,
        neighborhood_size: int | None = None,
        backend: str | None = None,
        device: str | None = None,
        implementation: str = "leidenalg",
        **kwargs,
    ):
        if implementation not in _IMPLEMENTATIONS:
            raise ValueError(
                f"implementation must be 'leidenalg' or 'sknetwork', got {implementation!r}"
            )
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device,
            neighborhood_size=neighborhood_size, **kwargs,
        )
        self.resolution = resolution
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.random_state = random_state
        self.implementation = implementation
        self._labels = None
        self._adjacency = None

    def _build_adjacency(self, x_np: np.ndarray, cache: dict | None = None) -> scipy.sparse.csr_matrix:
        """Build symmetrized kNN adjacency using the shared compute_knn cache."""
        from ...utils.metrics import compute_knn

        _, indices = compute_knn(x_np, k=self.n_neighbors, include_self=False, cache=cache)
        n = x_np.shape[0]
        k = indices.shape[1]
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        data = np.ones(len(rows), dtype=np.float32)
        adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        # Symmetrize
        adj = adj + adj.T
        adj.data[adj.data > 1] = 1
        return adj

    def _run_leiden(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """Run Leiden on a sparse adjacency matrix with the chosen implementation."""
        if self.implementation == "sknetwork":
            return self._run_sknetwork(adjacency)
        return self._run_leidenalg(adjacency)

    def _run_sknetwork(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """scikit-network (BSD) Leiden; imports neither igraph nor leidenalg.

        Not thread-safe: scikit-network 0.33.5's refinement step draws from libc `rand()`
        (`leiden_core.pyx`), which its `random_state` does not seed. This seeds libc with
        `srand(random_state)` right before the fit, and `srand` is process-global.
        """
        from sknetwork.clustering import Leiden

        if self.random_state is not None:
            _libc().srand(int(self.random_state) & 0xFFFFFFFF)
        model = Leiden(resolution=self.resolution, random_state=self.random_state)
        labels = model.fit_predict(scipy.sparse.csr_matrix(adjacency))
        return np.asarray(labels, dtype=np.int64).reshape(-1)

    def _run_leidenalg(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """leidenalg (GPL) Leiden on an igraph graph."""
        import igraph as ig
        import leidenalg

        adj_coo = scipy.sparse.coo_matrix(adjacency)
        edges = list(zip(adj_coo.row.tolist(), adj_coo.col.tolist()))
        weights = adj_coo.data.tolist()

        g = ig.Graph(n=adjacency.shape[0], edges=edges, directed=False)
        g.es["weight"] = weights
        g.simplify(combine_edges="max")

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_state,
            weights="weight",
        )
        return np.array(partition.membership, dtype=np.int64)

    def fit(self, x, y=None) -> None:
        x_np = _to_numpy(x)
        self._adjacency = self._build_adjacency(x_np)
        self._labels = self._run_leiden(self._adjacency)
        self._is_fitted = True
        self._remember_fit_input(x)

    def transform(self, x):
        """Cluster assignments for the fitted rows.

        Leiden is TRANSDUCTIVE: the partition is defined on the graph built at fit time, so
        there is no mapping for unseen points. Returning `self._labels` regardless of `x` — as
        this did — pairs fit-order labels with whatever rows the caller passed. `run_experiment`
        fits the (shuffled) train tensor and transforms the (unshuffled) test tensor, so the
        result was a silent permutation: ARI against ground truth -0.0012 instead of +1.0000,
        with the right shape and no error.

        Raising NotImplementedError is what the caller is already prepared for — it falls back
        to `fit_transform` on the array actually being embedded.
        """
        if not self._is_fitted:
            raise RuntimeError("LeidenModule is not fitted. Call fit() first.")
        if not self._same_as_fit_input(x):
            raise NotImplementedError(
                "LeidenModule is transductive: it cannot assign clusters to rows it was not "
                "fitted on. Call fit_transform(x) on the data you want labelled."
            )
        labels_np = self._labels.reshape(-1, 1).astype(np.float32)
        return _to_output(labels_np, x)

    def fit_from_graph(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """Run Leiden on a precomputed adjacency matrix.

        This is a convenience method for use outside the standard pipeline.
        """
        self._adjacency = adjacency
        self._labels = self._run_leiden(adjacency)
        self._is_fitted = True
        return self._labels

    def adjacency(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Return the binary kNN adjacency matrix as an (N, N) dense array."""
        if not self._is_fitted:
            raise RuntimeError("LeidenModule is not fitted. Call fit() first.")
        A = self._adjacency.toarray().astype(np.float64)
        A[A > 0] = 1.0
        if ignore_diagonal:
            np.fill_diagonal(A, 0)
        return A

    def kernel(self, ignore_diagonal: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("LeidenModule is not fitted. Call fit() first.")
        K = self._adjacency.toarray().astype(np.float64)
        if ignore_diagonal:
            np.fill_diagonal(K, 0)
        return K
