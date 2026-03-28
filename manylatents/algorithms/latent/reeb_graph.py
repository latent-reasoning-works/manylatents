"""Reeb graph approximation as a LatentModule."""
import logging

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .latent_module_base import LatentModule, _to_numpy, _to_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lens functions — map point cloud X → scalar per point
# ---------------------------------------------------------------------------

def _compute_lens_density(X, k=15, cache=None):
    """Inverse mean kNN distance — high where data is dense."""
    from ...utils.metrics import compute_knn

    dists, _ = compute_knn(X, k=k, include_self=False, cache=cache)
    mean_dist = dists.mean(axis=1)
    mean_dist[mean_dist == 0] = 1e-10
    return 1.0 / mean_dist


def _compute_lens_pca1(X):
    """First principal component."""
    return PCA(n_components=1).fit_transform(X).ravel()


def _compute_lens_diffusion1(X, k=15, t=1, cache=None):
    """First non-trivial diffusion coordinate."""
    from scipy.sparse import csr_matrix, diags
    from ...utils.metrics import compute_knn

    dists, indices = compute_knn(X, k=k, include_self=False, cache=cache)
    n = len(X)
    # adaptive bandwidth: distance to k-th neighbour
    sigma = dists[:, -1].copy()
    sigma[sigma == 0] = 1e-10
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(dists.shape[1]):
            j = indices[i, j_idx]
            w = np.exp(-dists[i, j_idx] ** 2 / (sigma[i] * sigma[j]))
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    # row-normalise → Markov matrix, power iterate
    D_inv = 1.0 / np.array(W.sum(axis=1)).ravel()
    D_inv[~np.isfinite(D_inv)] = 0
    P = diags(D_inv) @ W
    for _ in range(t - 1):
        P = P @ P
    eigvals, eigvecs = np.linalg.eigh(P.toarray())
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[1]]


_LENS_REGISTRY = {
    "density": _compute_lens_density,
    "pca1": _compute_lens_pca1,
    "diffusion1": _compute_lens_diffusion1,
}


# ---------------------------------------------------------------------------
# Vietoris-Rips and Reeb graph construction
# ---------------------------------------------------------------------------

def _vietoris_rips(D, min_rad_factor=1.5, max_dim=2, sparse=0.5, min_rad=None):
    from ripser import ripser
    import gudhi as gd

    if min_rad is None:
        min_rad = ripser(D, distance_matrix=True, maxdim=0)['dgms'][0][-2][1]
    radius = min_rad_factor * min_rad
    rips = gd.RipsComplex(distance_matrix=D, max_edge_length=radius, sparse=sparse)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    filt = st.get_filtration()
    zero_skel = [s[0][0] for s in filt if len(s[0]) == 1]
    one_skel = [s[0] for s in filt if len(s[0]) == 2]
    two_skel = [s[0] for s in filt if len(s[0]) == 3]
    return st, [zero_skel, one_skel, two_skel]


def _vr_to_graph(skeleta, function, with_weights=False):
    import networkx as nx

    nodes = skeleta[0]
    edges = skeleta[1]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    if with_weights:
        weighted_edges = [(u, v, np.abs(function[u] - function[v])) for u, v in edges]
        G.add_weighted_edges_from(weighted_edges)
    else:
        G.add_edges_from(edges)
    return G


def _reeb_approx_graph(H, function, n_bins, overlap=0.0, density_factor=0.1):
    """Build the approximate Reeb graph.

    Each Reeb node is a connected component within a (possibly overlapping)
    bin of the lens function.  A data point belongs to every Reeb node whose
    bin contains it **and** whose connected component it falls in.  With
    ``overlap > 0``, points near bin boundaries belong to multiple Reeb nodes.

    Parameters
    ----------
    overlap : float
        Fraction of bin width to extend on each side.  ``0.0`` gives strict
        non-overlapping bins; ``0.25`` extends each bin by 25 % of its width
        on both sides so adjacent bins share a region.

    Returns
    -------
    G : nx.Graph
        Reeb graph with ``indices`` attribute on each node.
    membership : np.ndarray, shape (N, M)
        Binary membership matrix — ``membership[i, j] == 1`` iff data point
        *i* belongs to Reeb node *j*.  Rows can sum to >1 when overlap > 0.
    """
    import networkx as nx

    n_points = len(function)
    f_min, f_max = np.min(function), np.max(function) + 1e-6
    bin_width = (f_max - f_min) / n_bins
    ext = overlap * bin_width  # extension on each side

    # Build (possibly overlapping) bin intervals
    edges_lo = np.array([f_min + i * bin_width - ext for i in range(n_bins)])
    edges_hi = np.array([f_min + (i + 1) * bin_width + ext for i in range(n_bins)])

    subsets_idx = [
        np.where((edges_lo[i] <= function) & (function < edges_hi[i]))[0]
        for i in range(n_bins)
    ]

    G = nx.Graph()
    node_counter = 0
    prev_components = []
    prev_node_ids = []
    # Track which Reeb nodes each point belongs to (multi-label)
    point_nodes: list[list[int]] = [[] for _ in range(n_points)]

    for j, subset in enumerate(subsets_idx):
        H_sub = H.subgraph(subset)
        comps = [
            list(c) for c in nx.connected_components(H_sub)
            if len(c) > n_points / n_bins * density_factor
        ]
        curr_node_ids = []
        for k, comp in enumerate(comps):
            G.add_node(node_counter, indices=comp)
            for pt in comp:
                point_nodes[pt].append(node_counter)
            # Connect to previous bin's components if they share VR-connectivity
            for prev_idx, prev in enumerate(prev_components):
                if nx.is_connected(H.subgraph(comp + prev)):
                    G.add_edge(node_counter, prev_node_ids[prev_idx])
            curr_node_ids.append(node_counter)
            node_counter += 1
        prev_components = comps
        prev_node_ids = curr_node_ids

    # Build (N, M) binary membership matrix
    m = G.number_of_nodes()
    membership = np.zeros((n_points, m), dtype=np.float32)
    for i, nodes in enumerate(point_nodes):
        for nid in nodes:
            membership[i, nid] = 1.0

    return G, membership


def _structural_summary(G) -> dict:
    """Extract topological invariants from a Reeb graph."""
    import networkx as nx

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "degree_sequence": sorted(degrees, reverse=True),
        "n_branch_points": sum(1 for d in degrees if d >= 3),
        "n_endpoints": sum(1 for d in degrees if d == 1),
        "n_loops": n_edges - n_nodes + n_components,  # Euler formula
    }


# ---------------------------------------------------------------------------
# LatentModule
# ---------------------------------------------------------------------------

class ReebGraphModule(LatentModule):
    """Approximate Reeb graph as a LatentModule.

    Computes a decorated Reeb graph on input data and returns an (N, M)
    binary membership matrix where M is the number of Reeb nodes. Each row
    indicates which Reeb nodes a data point belongs to.

    With ``overlap=0.0`` (strict), each point belongs to exactly one node.
    With ``overlap>0`` (overlapping covers), points near bin boundaries
    belong to multiple nodes.

    The Reeb graph adjacency is exposed via ``adjacency_matrix()`` and node
    positions in the input space via ``node_coordinates``.

    Parameters
    ----------
    n_bins : int
        Number of bins for the Reeb graph approximation.
    overlap : float
        Fraction of bin width to extend on each side. ``0.0`` gives strict
        non-overlapping bins (classical Reeb); ``0.25`` gives 25 % overlap
        (Mapper-style soft assignment).
    lens : str
        Lens function: ``"default"`` (first coordinate), ``"density"``,
        ``"pca1"``, or ``"diffusion1"``.
    lens_k : int
        Neighbourhood size for density / diffusion lenses.
    lens_t : int
        Diffusion time for the diffusion1 lens.
    sparse : float
        Sparsity parameter for the Vietoris-Rips complex.
    min_rad_factor : float
        Multiplier on the minimum radius for the VR complex.
    density_factor : float
        Minimum component size as a fraction of ``n_points / n_bins``.
    """

    def __init__(
        self,
        n_components: int = 1,
        n_bins: int = 10,
        overlap: float = 0.25,
        lens: str = "default",
        lens_k: int = 15,
        lens_t: int = 1,
        sparse: float = 0.5,
        min_rad_factor: float = 2,
        density_factor: float = 0.0,
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
        self.n_bins = n_bins
        self.overlap = overlap
        self.lens = lens
        self.lens_k = neighborhood_size if neighborhood_size is not None else lens_k
        self.lens_t = lens_t
        self.sparse = sparse
        self.min_rad_factor = min_rad_factor
        self.density_factor = density_factor
        self.random_state = random_state

        self._reeb_graph = None
        self._membership = None  # (N, M) binary membership matrix
        self._adjacency = None
        self.structural_summary = None
        self.node_coordinates = None  # (M, D) mean coordinates per Reeb node

    def _compute_lens(self, x_np: np.ndarray) -> np.ndarray:
        """Compute the lens function on input data."""
        import inspect

        if self.lens == "default":
            return x_np[:, 0]
        elif self.lens in _LENS_REGISTRY:
            fn = _LENS_REGISTRY[self.lens]
            sig = inspect.signature(fn)
            kwargs = {}
            if "k" in sig.parameters:
                kwargs["k"] = self.lens_k
            if "t" in sig.parameters:
                kwargs["t"] = self.lens_t
            if "cache" in sig.parameters:
                kwargs["cache"] = None
            return fn(x_np, **kwargs)
        else:
            raise ValueError(
                f"Unknown lens '{self.lens}'. Choose from: default, "
                f"{', '.join(_LENS_REGISTRY)}"
            )

    def fit(self, x, y=None) -> None:
        from ripser import ripser  # noqa: F401
        import gudhi  # noqa: F401
        import networkx as nx

        x_np = _to_numpy(x)

        # Compute lens function
        function = self._compute_lens(x_np)

        # Build distance matrix and VR complex
        D = pairwise_distances(x_np)
        min_rad = ripser(D, distance_matrix=True, maxdim=0)['dgms'][0][-2][1]
        _, skeleta = _vietoris_rips(
            D, min_rad_factor=self.min_rad_factor,
            sparse=self.sparse, min_rad=min_rad,
        )
        H = _vr_to_graph(skeleta, function, with_weights=True)

        # Approximate Reeb graph
        G, membership = _reeb_approx_graph(
            H, function, self.n_bins,
            overlap=self.overlap, density_factor=self.density_factor,
        )

        self._reeb_graph = G
        self._membership = membership

        # Build adjacency matrix (M x M, where M = number of Reeb nodes)
        # and compute mean coordinates per Reeb node in the input space
        m = G.number_of_nodes()
        node_list = sorted(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        adj = np.zeros((m, m), dtype=np.float64)
        for u, v in G.edges():
            adj[node_to_idx[u], node_to_idx[v]] = 1.0
            adj[node_to_idx[v], node_to_idx[u]] = 1.0
        self._adjacency = adj

        # Mean coordinate of data points assigned to each Reeb node
        d = x_np.shape[1]
        coords = np.zeros((m, d), dtype=np.float64)
        for node_id in node_list:
            idx = node_to_idx[node_id]
            pt_indices = G.nodes[node_id].get("indices", [])
            if len(pt_indices) > 0:
                coords[idx] = x_np[pt_indices].mean(axis=0)
        self.node_coordinates = coords
        # Also store as pos on the networkx graph for convenience
        nx.set_node_attributes(
            G, {n: coords[node_to_idx[n]] for n in node_list}, "pos"
        )

        self.structural_summary = _structural_summary(G)
        self._is_fitted = True

        logger.info(
            f"ReebGraphModule(lens={self.lens}, overlap={self.overlap}): "
            f"{self.structural_summary['n_nodes']} nodes, "
            f"{self.structural_summary['n_edges']} edges, "
            f"{self.structural_summary['n_branch_points']} branch pts"
        )

    def transform(self, x):
        """Return (N, M) binary membership matrix. Matches input type."""
        if not self._is_fitted:
            raise RuntimeError("ReebGraphModule is not fitted. Call fit() first.")
        return _to_output(self._membership, x)

    def adjacency(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Return the Reeb graph adjacency as an (M, M) binary dense array.

        M is the number of Reeb nodes, not the number of data points.
        """
        if not self._is_fitted:
            raise RuntimeError("ReebGraphModule is not fitted. Call fit() first.")
        A = self._adjacency.copy()
        if ignore_diagonal:
            np.fill_diagonal(A, 0)
        return A

    def extra_outputs(self) -> dict:
        """Collect ReebGraph-specific outputs in addition to base outputs."""
        extras = super().extra_outputs()
        if getattr(self, "node_coordinates", None) is not None:
            extras["node_coordinates"] = self.node_coordinates
        if getattr(self, "structural_summary", None) is not None:
            extras["structural_summary"] = self.structural_summary
        return extras
