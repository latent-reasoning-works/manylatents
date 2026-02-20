"""Leiden community detection for embedding analysis."""
import numpy as np
import scipy.sparse


class LeidenClustering:
    """Cluster embeddings or kNN graphs using the Leiden algorithm.

    Parameters
    ----------
    resolution : float
        Leiden resolution parameter. Higher values produce more clusters.
    n_neighbors : int
        Number of neighbors for kNN graph construction (used by fit()).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, resolution: float = 0.5, n_neighbors: int = 15, random_state: int = 42):
        self.resolution = resolution
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, embedding: np.ndarray) -> np.ndarray:
        """Build kNN graph on embedding, run Leiden, return cluster labels."""
        from sklearn.neighbors import kneighbors_graph

        adj = kneighbors_graph(
            embedding, n_neighbors=self.n_neighbors, mode="connectivity", include_self=False
        )
        # Symmetrize the adjacency matrix
        adj = adj + adj.T
        adj[adj > 1] = 1
        return self.fit_from_graph(adj)

    def fit_from_graph(self, adjacency: scipy.sparse.spmatrix) -> np.ndarray:
        """Run Leiden on a precomputed adjacency matrix."""
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
