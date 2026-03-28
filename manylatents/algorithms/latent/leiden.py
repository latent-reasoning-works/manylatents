"""Leiden community detection as a LatentModule."""
import numpy as np
import scipy.sparse
import torch
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output


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
    """

    def __init__(
        self,
        n_components: int = 1,
        resolution: float = 0.5,
        n_neighbors: int = 15,
        random_state: int = 42,
        neighborhood_size: int | None = None,
        backend: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device,
            neighborhood_size=neighborhood_size, **kwargs,
        )
        self.resolution = resolution
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.random_state = random_state
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
        """Run Leiden on a sparse adjacency matrix."""
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

    def transform(self, x):
        if not self._is_fitted:
            raise RuntimeError("LeidenModule is not fitted. Call fit() first.")
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
