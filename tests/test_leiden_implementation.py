"""LeidenModule's `implementation` option: leidenalg (GPL, the default) or sknetwork (BSD)."""
import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from manylatents.algorithms.latent.leiden import LeidenModule

pytest.importorskip("sknetwork")


def _blobs(n_per_cluster=50, n_clusters=2, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 20
    data = np.vstack([centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
                      for i in range(n_clusters)])
    truth = np.repeat(np.arange(n_clusters), n_per_cluster)
    return torch.from_numpy(data).float(), truth


def _leidenalg_reference(adjacency, resolution, seed):
    """The leidenalg call that LeidenModule made before the option existed."""
    import igraph as ig
    import leidenalg
    import scipy.sparse

    coo = scipy.sparse.coo_matrix(adjacency)
    g = ig.Graph(n=adjacency.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())),
                 directed=False)
    g.es["weight"] = coo.data.tolist()
    g.simplify(combine_edges="max")
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed, weights="weight")
    return np.array(part.membership, dtype=np.int64)


def test_default_is_leidenalg_and_labels_are_unchanged():
    pytest.importorskip("leidenalg")
    data, _ = _blobs(n_clusters=5, n_per_cluster=40)
    default = LeidenModule(resolution=0.5, n_neighbors=10, random_state=42)
    assert default.implementation == "leidenalg"
    explicit = LeidenModule(resolution=0.5, n_neighbors=10, random_state=42,
                            implementation="leidenalg")
    labels_default = default.fit_transform(data).squeeze().numpy().astype(np.int64)
    labels_explicit = explicit.fit_transform(data).squeeze().numpy().astype(np.int64)
    np.testing.assert_array_equal(labels_default, labels_explicit)
    reference = _leidenalg_reference(default._adjacency, 0.5, 42)
    np.testing.assert_array_equal(labels_default, reference)


def test_sknetwork_recovers_two_blobs():
    data, truth = _blobs()
    lm = LeidenModule(resolution=0.5, n_neighbors=10, random_state=42, implementation="sknetwork")
    lm.fit(data)
    labels = lm._labels
    assert labels.dtype == np.int64
    assert labels.shape == (data.shape[0],)
    assert adjusted_rand_score(truth, labels) == 1.0


def test_sknetwork_is_deterministic_for_one_random_state():
    data, _ = _blobs(n_clusters=5, n_per_cluster=40)
    a = LeidenModule(random_state=7, implementation="sknetwork").fit_transform(data)
    b = LeidenModule(random_state=7, implementation="sknetwork").fit_transform(data)
    torch.testing.assert_close(a, b)


def test_sknetwork_repeats_on_a_graph_with_ties():
    # scikit-network 0.33.5 draws from libc rand() in its refinement step, which its
    # random_state does not seed. On structureless data that draw changes the partition.
    data = np.random.default_rng(0).normal(size=(2000, 10)).astype(np.float32)
    lm = LeidenModule(resolution=0.5, n_neighbors=15, random_state=42, implementation="sknetwork")
    adjacency = lm._build_adjacency(data)
    first = lm._run_leiden(adjacency)
    for _ in range(4):
        np.testing.assert_array_equal(lm._run_leiden(adjacency), first)


def test_sknetwork_works_without_igraph_and_leidenalg(monkeypatch):
    monkeypatch.setitem(sys.modules, "igraph", None)
    monkeypatch.setitem(sys.modules, "leidenalg", None)
    data, truth = _blobs()
    lm = LeidenModule(resolution=0.5, n_neighbors=10, random_state=42, implementation="sknetwork")
    lm.fit(data)
    assert adjusted_rand_score(truth, lm._labels) == 1.0


def test_bad_implementation_raises_at_construction():
    with pytest.raises(ValueError, match="leidenalg.*sknetwork"):
        LeidenModule(implementation="networkx")
