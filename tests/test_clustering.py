"""Tests for LeidenClustering analysis module."""
import numpy as np
import pytest

leidenalg = pytest.importorskip("leidenalg")


def _make_blobs(n_per_cluster=100, n_clusters=5, n_features=10, seed=42):
    """Generate well-separated Gaussian blobs."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 10
    data = np.vstack([
        centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
        for i in range(n_clusters)
    ])
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return data, labels


class TestLeidenClustering:
    def test_fit_returns_labels(self):
        """fit() returns integer cluster labels with correct shape."""
        from manylatents.analysis.clustering import LeidenClustering
        data, _ = _make_blobs()
        lc = LeidenClustering(resolution=0.5, n_neighbors=15, random_state=42)
        labels = lc.fit(data)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (data.shape[0],)
        assert labels.dtype in (np.int32, np.int64)

    def test_finds_approximately_correct_clusters(self):
        """On well-separated blobs, Leiden finds ~5 clusters."""
        from manylatents.analysis.clustering import LeidenClustering
        data, _ = _make_blobs(n_clusters=5)
        lc = LeidenClustering(resolution=0.5, n_neighbors=15, random_state=42)
        labels = lc.fit(data)
        n_clusters = len(np.unique(labels))
        assert 3 <= n_clusters <= 8, f"Expected ~5 clusters, got {n_clusters}"

    def test_fit_from_graph(self):
        """fit_from_graph() accepts a sparse adjacency matrix."""
        from manylatents.analysis.clustering import LeidenClustering
        from sklearn.neighbors import kneighbors_graph
        data, _ = _make_blobs()
        adj = kneighbors_graph(data, n_neighbors=15, mode="connectivity")
        lc = LeidenClustering(random_state=42)
        labels = lc.fit_from_graph(adj)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (data.shape[0],)

    def test_deterministic(self):
        """Same random_state produces same labels."""
        from manylatents.analysis.clustering import LeidenClustering
        data, _ = _make_blobs()
        lc1 = LeidenClustering(random_state=42)
        lc2 = LeidenClustering(random_state=42)
        labels1 = lc1.fit(data)
        labels2 = lc2.fit(data)
        np.testing.assert_array_equal(labels1, labels2)

    def test_resolution_affects_cluster_count(self):
        """Higher resolution produces more clusters."""
        from manylatents.analysis.clustering import LeidenClustering
        data, _ = _make_blobs(n_clusters=5)
        lc_low = LeidenClustering(resolution=0.1, n_neighbors=15, random_state=42)
        lc_high = LeidenClustering(resolution=2.0, n_neighbors=15, random_state=42)
        n_low = len(np.unique(lc_low.fit(data)))
        n_high = len(np.unique(lc_high.fit(data)))
        assert n_high >= n_low, f"High res ({n_high}) should have >= clusters than low res ({n_low})"
