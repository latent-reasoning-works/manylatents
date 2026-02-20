"""Tests for LeidenModule (LatentModule interface)."""
import numpy as np
import pytest
import torch

leidenalg = pytest.importorskip("leidenalg")


def _make_blobs(n_per_cluster=100, n_clusters=5, n_features=10, seed=42):
    """Generate well-separated Gaussian blobs."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 10
    data = np.vstack([
        centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
        for i in range(n_clusters)
    ])
    return data


class TestLeidenModule:
    def test_is_latent_module(self):
        """LeidenModule is a LatentModule subclass."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        from manylatents.algorithms.latent.latent_module_base import LatentModule
        lm = LeidenModule(resolution=0.5)
        assert isinstance(lm, LatentModule)

    def test_fit_transform_returns_tensor(self):
        """fit_transform returns (N, 1) float tensor of cluster labels."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        data = torch.from_numpy(_make_blobs()).float()
        lm = LeidenModule(resolution=0.5, n_neighbors=15, random_state=42)
        result = lm.fit_transform(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (data.shape[0], 1)

    def test_finds_approximately_correct_clusters(self):
        """On well-separated blobs, Leiden finds ~5 clusters."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        data = torch.from_numpy(_make_blobs(n_clusters=5)).float()
        lm = LeidenModule(resolution=0.5, n_neighbors=15, random_state=42)
        result = lm.fit_transform(data)
        labels = result.squeeze().numpy().astype(int)
        n_clusters = len(np.unique(labels))
        assert 3 <= n_clusters <= 8, f"Expected ~5 clusters, got {n_clusters}"

    def test_fit_from_graph(self):
        """fit_from_graph() accepts a sparse adjacency matrix."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        from sklearn.neighbors import kneighbors_graph
        data = _make_blobs()
        adj = kneighbors_graph(data, n_neighbors=15, mode="connectivity")
        lm = LeidenModule(random_state=42)
        labels = lm.fit_from_graph(adj)
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (data.shape[0],)

    def test_deterministic(self):
        """Same random_state produces same labels."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        data = torch.from_numpy(_make_blobs()).float()
        lm1 = LeidenModule(random_state=42)
        lm2 = LeidenModule(random_state=42)
        r1 = lm1.fit_transform(data)
        r2 = lm2.fit_transform(data)
        torch.testing.assert_close(r1, r2)

    def test_resolution_affects_cluster_count(self):
        """Higher resolution produces more clusters."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        data = torch.from_numpy(_make_blobs(n_clusters=5)).float()
        lm_low = LeidenModule(resolution=0.1, n_neighbors=15, random_state=42)
        lm_high = LeidenModule(resolution=2.0, n_neighbors=15, random_state=42)
        n_low = len(torch.unique(lm_low.fit_transform(data)))
        n_high = len(torch.unique(lm_high.fit_transform(data)))
        assert n_high >= n_low

    def test_kernel_matrix(self):
        """kernel_matrix() returns the kNN adjacency."""
        from manylatents.algorithms.latent.leiden import LeidenModule
        data = torch.from_numpy(_make_blobs(n_per_cluster=20)).float()
        lm = LeidenModule(n_neighbors=10, random_state=42)
        lm.fit(data)
        K = lm.kernel_matrix()
        n = data.shape[0]
        assert K.shape == (n, n)
        assert np.allclose(K, K.T)  # symmetric
