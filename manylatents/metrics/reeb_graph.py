import logging
from typing import Optional, Dict, List

import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)

class DecoratedReebGraph:
    def __init__(self, data, function=None, distance_matrix=False):
        from ripser import ripser

        self.data = data
        self.function = function if function is not None else data[:, 0]
        self.distance_matrix = distance_matrix

        if distance_matrix:
            self.distances = data
        else:
            self.distances = pairwise_distances(data)

        min_rad = ripser(self.distances, distance_matrix=True, maxdim=0)['dgms'][0][-2][1]
        self.metadata = {'Connected VR complex radius': min_rad}

        if not distance_matrix:
            if data.shape[1] in [2, 3]:
                self.coords = data
            else:
                self.coords = PCA(n_components=3).fit_transform(data)

        self.VRComplex = None
        self.VRSkeleta = None
        self.VRGraph = None
        self.ReebGraph = None

    def fit_Vietoris_Rips(self, sparse=0.5, min_rad_factor=2, max_dim=2, with_weights=True):
        D = self.distances
        min_rad = self.metadata['Connected VR complex radius']
        Rips_complex, skeleta = Vietoris_Rips(D, min_rad_factor, max_dim, sparse, min_rad)
        H = VR_to_graph(skeleta, self.function, with_weights)
        self.VRComplex = Rips_complex
        self.VRSkeleta = skeleta
        self.VRGraph = H

    def fit_Reeb(self, n_bins=10, density_factor=0.0, add_coords=True):
        if self.VRComplex is None:
            self.fit_Vietoris_Rips()
        H = self.VRGraph
        G, _ = Reeb_approx_graph(H, self.function, n_bins, return_embedding_data=True, density_factor=density_factor)
        self.ReebGraph = G



def Vietoris_Rips(D, min_rad_factor=1.5, max_dim=2, sparse=0.5, min_rad=None):
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

def VR_to_graph(skeleta, function, with_weights=False):
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

def Reeb_approx_graph(H, function, n_bins, return_embedding_data=True, density_factor=0.1):
    bin_range = [np.min(function), np.max(function) + 1e-6]
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    subsets_idx = [np.where((bins[i] <= function) & (function < bins[i + 1]))[0] for i in range(n_bins)]

    G = nx.Graph()
    attrs = {}
    colors = []
    node_counter = 0
    prev_components = []

    for j, subset in enumerate(subsets_idx):
        H_sub = H.subgraph(subset)
        comps = [list(c) for c in nx.connected_components(H_sub) if len(c) > len(H) / n_bins * density_factor]
        for k, comp in enumerate(comps):
            node = (j, k)
            G.add_node(node, indices=comp)
            attrs[node] = {'component indices': comp}
            if return_embedding_data:
                colors.append((bins[j] + bins[j + 1]) / 2)
            for ll, prev in enumerate(prev_components):
                if nx.is_connected(H.subgraph(comp + prev)):
                    G.add_edge(node, (j - 1, ll))
        prev_components = comps
    nx.set_node_attributes(G, attrs)
    return G, colors


@register_metric(
    aliases=["reeb_graph"],
    default_params={"n_bins": 10},
    description="Reeb graph node and edge counts",
)
def ReebGraphNodesEdges(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    n_bins: int = 10,
    cache: Optional[dict] = None,
) -> Dict[str, List]:
    """
    Compute a decorated Reeb graph on `embeddings` and return lists of nodes and edges.

    Returns:
      dict with keys 'nodes' and 'edges'.
    """
    try:
        from ripser import ripser  # noqa: F401
        import gudhi  # noqa: F401
    except ImportError:
        import warnings
        warnings.warn("ripser/gudhi not installed — ReebGraphNodesEdges returning empty", RuntimeWarning)
        return {'nodes': [], 'edges': []}

    drg = DecoratedReebGraph(data=embeddings, function=embeddings[:, 0])
    drg.fit_Reeb(n_bins=n_bins)

    G = drg.ReebGraph
    nodes = list(G.nodes())
    edges = list(G.edges())

    logger.info(f"ReebGraphNodesEdges computed with {len(nodes)} nodes and {len(edges)} edges.")

    return {'nodes': nodes, 'edges': edges}



__all__ = ['ReebGraphNodesEdges']
