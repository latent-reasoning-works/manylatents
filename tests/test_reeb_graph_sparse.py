"""Exact comparisons with the fit path before the bounded-memory rewrite."""
import json
import os
import subprocess
import sys
import random
import logging

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll
from sklearn.metrics import pairwise_distances
from ripser import ripser
from manylatents.algorithms.latent import reeb_graph as rg
from manylatents.algorithms.latent.reeb_graph import (
    _vietoris_rips, _vr_to_graph, _reeb_approx_graph, _structural_summary,
)
from manylatents.algorithms.latent.latent_module_base import _to_numpy
logger = logging.getLogger(__name__)

def _reference_fit(self, x, y=None) -> None:
    from ripser import ripser  # noqa: F401
    import gudhi  # noqa: F401
    import networkx as nx

    x_np = _to_numpy(x)
    n = x_np.shape[0]

    # Cell cap: the same draw as PersistentHomology, on a local Random instance.
    subsampled = self.max_N is not None and n > self.max_N
    if subsampled:
        subsample_indices = np.array(
            random.Random(self.random_state).sample(range(n), k=self.max_N),
            dtype=np.int64,
        )
        logger.info(f"ReebGraphModule: subsampled {n} → {self.max_N}")
        x_sub = x_np[subsample_indices]
    else:
        subsample_indices = np.arange(n, dtype=np.int64)
        x_sub = x_np
    self.subsample_indices = subsample_indices

    # Compute lens function
    function = self._compute_lens(x_sub)

    # Build only vertices and edges: _vr_to_graph does not use triangles,
    # which can consume cubic memory when the radius connects all points.
    D = pairwise_distances(x_sub)
    min_rad = ripser(D, distance_matrix=True, maxdim=0)['dgms'][0][-2][1]
    _, skeleta = _vietoris_rips(
        D, min_rad_factor=self.min_rad_factor, max_dim=1,
        sparse=self.sparse, min_rad=min_rad,
    )
    H = _vr_to_graph(skeleta, function, with_weights=True)

    # Approximate Reeb graph
    G, membership = _reeb_approx_graph(
        H, function, self.n_bins,
        overlap=self.overlap, density_factor=self.density_factor,
    )

    self._reeb_graph = G
    if subsampled:
        # (max_N, M) -> (N, M): each row takes its nearest sampled row's membership,
        # then each sampled row is reset to its own (duplicate rows stay distinct).
        from sklearn.neighbors import NearestNeighbors

        nn_idx = (
            NearestNeighbors(n_neighbors=1)
            .fit(x_sub)
            .kneighbors(x_np, return_distance=False)
        )
        full = membership[nn_idx[:, 0]]
        full[subsample_indices] = membership
        membership = full
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

    # Mean coordinate of data points assigned to each Reeb node. The node
    # "indices" index into x_sub, the rows the graph was built on.
    d = x_sub.shape[1]
    coords = np.zeros((m, d), dtype=np.float64)
    for node_id in node_list:
        idx = node_to_idx[node_id]
        pt_indices = G.nodes[node_id].get("indices", [])
        if len(pt_indices) > 0:
            coords[idx] = x_sub[pt_indices].mean(axis=0)
    self.node_coordinates = coords
    # Also store as pos on the networkx graph for convenience
    nx.set_node_attributes(
        G, {n: coords[node_to_idx[n]] for n in node_list}, "pos"
    )

    self.structural_summary = _structural_summary(G)
    self._is_fitted = True
    self._remember_fit_input(x)

    logger.info(
        f"ReebGraphModule(lens={self.lens}, overlap={self.overlap}): "
        f"{self.structural_summary['n_nodes']} nodes, "
        f"{self.structural_summary['n_edges']} edges, "
        f"{self.structural_summary['n_branch_points']} branch pts"
    )


def data_case(name, n=360):
    rng = np.random.default_rng(17)
    t = np.linspace(-1, 1, n)
    if name == "swissroll":
        return make_swiss_roll(n_samples=n, random_state=17)[0]
    if name == "curve":
        return np.column_stack((t, t**2, t**3))
    if name == "tree":
        arm = np.arange(n) % 3
        r = (np.arange(n) // 3 + 1) / (n / 3)
        angles = arm * 2 * np.pi / 3
        return np.column_stack((r * np.cos(angles), r * np.sin(angles)))
    if name == "circle":
        theta = t * np.pi
        return np.column_stack((np.cos(theta), np.sin(theta))) + rng.normal(0, .02, (n, 2))
    return make_blobs(n_samples=n, centers=[[-10, 0], [0, 10], [10, 0]],
                      cluster_std=.2, random_state=17)[0]


@pytest.mark.parametrize("name", ["swissroll", "curve", "tree", "circle", "blobs"])
@pytest.mark.parametrize("params", [{}, {"overlap": 0}, {"overlap": .25, "density_factor": .1},
                                    {"overlap": .25, "density_factor": 1.2}, {"max_N": 300}])
def test_reference_fit(name, params):
    x = data_case(name)
    old, new = rg.ReebGraphModule(**params), rg.ReebGraphModule(**params)
    _reference_fit(old, x)
    new.fit(x)
    assert list(old._reeb_graph) == list(new._reeb_graph)
    assert set(old._reeb_graph.edges) == set(new._reeb_graph.edges)
    for node in old._reeb_graph:
        assert set(old._reeb_graph.nodes[node]["indices"]) == set(new._reeb_graph.nodes[node]["indices"])
    np.testing.assert_array_equal(old._membership, new._membership)
    np.testing.assert_array_equal(old.transform(x), new.transform(x))
    np.testing.assert_array_equal(old._adjacency, new._adjacency)
    np.testing.assert_allclose(old.node_coordinates, new.node_coordinates, atol=1e-14)
    assert old.structural_summary == new.structural_summary
    assert old.transform(x).shape[1] == new.transform(x).shape[1]


@pytest.mark.parametrize("name", ["swissroll", "curve", "tree", "circle", "blobs"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_radius_exact(name, dtype):
    d = pairwise_distances(data_case(name).astype(dtype))
    assert rg._minimum_radius(d) == ripser(d, distance_matrix=True, maxdim=0)["dgms"][0][-2, 1]


@pytest.mark.parametrize("x", [np.array([[0.], [0.], [1e-12], [2e-12]]),
                                  np.array([[0.], [1.], [2.00000006]])])
def test_radius_small_and_rounding(x):
    d = pairwise_distances(x)
    assert rg._minimum_radius(d) == ripser(d, distance_matrix=True, maxdim=0)["dgms"][0][-2, 1]


def test_fit_without_complex(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fit must not build a global edge graph or call ripser")
    monkeypatch.setattr(rg, "_vietoris_rips", forbidden)
    monkeypatch.setattr("ripser.ripser", forbidden)
    rg.ReebGraphModule().fit(data_case("blobs"))


@pytest.mark.memory
def test_memory_blobs():
    script = """
import json, platform, resource, time
from sklearn.datasets import make_blobs
from manylatents.algorithms.latent.reeb_graph import ReebGraphModule
x, _ = make_blobs(n_samples=5000, centers=[[-10,0], [0,10], [10,0]], cluster_std=.2, random_state=17)
scale = 2**20 if platform.system() == 'Darwin' else 1024
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
start = time.perf_counter()
m = ReebGraphModule()
m.fit(x)
print(json.dumps(dict(seconds=time.perf_counter()-start, peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/scale, base_mib=base, nodes=m._membership.shape[1])))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    print(json.dumps(stats))
    # Includes imports plus the 191 MiB float64 distance matrix; generous host margin.
    assert stats["peak_mib"] < 1200


def test_filtration_and_component_order():
    # Zero-distance ties interleave edges and vertices in the filtration.
    x = np.zeros((360, 2))
    x[:, 0] = np.arange(360) % 37
    d = pairwise_distances(x)
    _, skeleta = _vietoris_rips(d, max_dim=1, min_rad=1)
    assert skeleta[0] == list(range(len(x)))
    h = _vr_to_graph(skeleta, x[:, 0])
    import networkx as nx
    for subset in [np.array([1, 33, 65, 97, 129, 161]), np.arange(300)]:
        expected = [set(c) for c in nx.connected_components(h.subgraph(subset))]
        actual = rg._bin_components(d, subset, 1.5, nx.empty_graph(len(x)))
        assert [set(c) for c in actual] == expected


def test_radius_boundary_uses_original_lower_triangle():
    d = np.array([[0., 1., 4.], [np.nextafter(1., np.inf), 0., 2.], [4., 2., 0.]])
    _, sk = _vietoris_rips(d, min_rad=1., min_rad_factor=1., max_dim=1)
    expected = {tuple(edge) for edge in sk[1]}
    mask = rg._radius_block(d, np.arange(3), np.arange(3), 1.)
    actual = {(i, j) for i in range(3) for j in range(i + 1, 3) if mask[i, j]}
    assert actual == expected == set()


def test_radius_all_duplicate_behavior():
    d = np.zeros((3, 3))
    with pytest.raises(IndexError):
        ripser(d, distance_matrix=True, maxdim=0)["dgms"][0][-2, 1]
    with pytest.raises(IndexError):
        rg._minimum_radius(d)
