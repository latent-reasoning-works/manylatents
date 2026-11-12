"""Tests for ReebGraphModule (LatentModule interface)."""
import numpy as np
import pytest
import torch

ripser = pytest.importorskip("ripser")
gudhi = pytest.importorskip("gudhi")


def _make_swissroll(n=200, seed=42):
    """Generate a simple Swiss roll dataset."""
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n))
    x = t * np.cos(t)
    y = 30 * rng.rand(n)
    z = t * np.sin(t)
    return np.column_stack([x, y, z])


class TestReebGraphModule:
    def test_fit_builds_no_triangles(self, monkeypatch):
        """Fit requests only vertices and edges, even for separated blobs."""
        from sklearn.datasets import make_blobs
        from manylatents.algorithms.latent import reeb_graph

        data, _ = make_blobs(
            n_samples=30, centers=[[-10, 0], [0, 10], [10, 0]],
            cluster_std=0.2, random_state=42,
        )
        real_vietoris_rips = reeb_graph._vietoris_rips
        calls = []

        def record_vietoris_rips(*args, **kwargs):
            tree, skeleta = real_vietoris_rips(*args, **kwargs)
            calls.append((kwargs.get("max_dim", 2), tree.dimension(), len(skeleta[2])))
            return tree, skeleta

        monkeypatch.setattr(reeb_graph, "_vietoris_rips", record_vietoris_rips)
        reeb_graph.ReebGraphModule(n_bins=5).fit(data)

        assert calls == [(1, 1, 0)]

    def test_one_skeleton_preserves_skeleta_and_fitted_graph(self, monkeypatch):
        """Removing triangles preserves vertices, edges, and fitted membership."""
        from sklearn.metrics import pairwise_distances
        from manylatents.algorithms.latent import reeb_graph

        rng = np.random.default_rng(42)
        theta = np.linspace(0, 2 * np.pi, 150, endpoint=False)
        data = np.column_stack([np.cos(theta), np.sin(theta)])
        data += rng.normal(scale=0.02, size=data.shape)
        distances = pairwise_distances(data)
        real_vietoris_rips = reeb_graph._vietoris_rips
        _, one = real_vietoris_rips(distances, min_rad_factor=2, max_dim=1)
        _, two = real_vietoris_rips(distances, min_rad_factor=2, max_dim=2)
        assert one[0] == two[0]
        assert one[1] == two[1]
        assert one[2] == []
        assert two[2]  # Ensure the comparison actually removes triangles.

        def fit_with_dimension(max_dim):
            def force_dimension(*args, **kwargs):
                kwargs["max_dim"] = max_dim
                return real_vietoris_rips(*args, **kwargs)

            monkeypatch.setattr(reeb_graph, "_vietoris_rips", force_dimension)
            model = reeb_graph.ReebGraphModule(n_bins=10, overlap=0.25)
            model.fit(data)
            return model

        one_model = fit_with_dimension(1)
        two_model = fit_with_dimension(2)
        assert set(one_model._reeb_graph.nodes) == set(two_model._reeb_graph.nodes)
        assert set(one_model._reeb_graph.edges) == set(two_model._reeb_graph.edges)
        for node in one_model._reeb_graph:
            assert set(one_model._reeb_graph.nodes[node]["indices"]) == set(
                two_model._reeb_graph.nodes[node]["indices"]
            )
        np.testing.assert_array_equal(one_model.transform(data), two_model.transform(data))

    def test_is_latent_module(self):
        """ReebGraphModule is a LatentModule subclass."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule
        from manylatents.algorithms.latent.latent_module_base import LatentModule

        m = ReebGraphModule()
        assert isinstance(m, LatentModule)

    def test_fit_transform_shape(self):
        """fit_transform returns (N, M) membership matrix."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        result = m.fit_transform(data)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == data.shape[0]  # N rows
        assert result.shape[1] > 0  # M Reeb nodes
        assert result.shape[1] == m.adjacency_matrix().shape[0]  # M matches adjacency

    def test_strict_single_membership(self):
        """With overlap=0.0, each point belongs to at most one Reeb node."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, overlap=0.0, random_state=42)
        result = m.fit_transform(data)
        memberships_per_point = result.sum(dim=1)
        assert (memberships_per_point <= 1).all()

    def test_overlap_multi_membership(self):
        """With overlap>0, some points belong to multiple Reeb nodes."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, overlap=0.5, random_state=42)
        result = m.fit_transform(data)
        memberships_per_point = result.sum(dim=1)
        # Some points should be in multiple Reeb nodes
        assert (memberships_per_point > 1).any(), "Expected multi-membership with overlap=0.5"

    def test_membership_is_binary(self):
        """Membership matrix values are 0 or 1."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, overlap=0.25, random_state=42)
        result = m.fit_transform(data)
        unique_vals = set(torch.unique(result).numpy().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_adjacency_matrix(self):
        """adjacency_matrix() returns a valid (M, M) symmetric adjacency."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        K = m.adjacency_matrix()
        assert K.ndim == 2
        assert K.shape[0] == K.shape[1]
        assert K.shape[0] > 0  # at least one Reeb node
        assert np.allclose(K, K.T)  # symmetric
        # Values are 0 or 1 (unweighted adjacency)
        assert set(np.unique(K)).issubset({0.0, 1.0})

    def test_structural_summary(self):
        """structural_summary has expected keys after fitting."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        s = m.structural_summary
        assert s is not None
        expected_keys = {
            "n_nodes", "n_edges", "n_components",
            "degree_sequence", "n_branch_points", "n_endpoints", "n_loops",
        }
        assert expected_keys == set(s.keys())
        assert s["n_nodes"] > 0

    def test_node_coordinates(self):
        """node_coordinates has shape (M, D) matching adjacency_matrix size."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        A = m.adjacency_matrix()
        coords = m.node_coordinates
        assert coords is not None
        assert coords.shape[0] == A.shape[0]  # M nodes
        assert coords.shape[1] == data.shape[1]  # D features
        # Coordinates should be finite
        assert np.all(np.isfinite(coords))

    def test_lens_density(self):
        """Density lens produces valid output."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, lens="density", lens_k=10, random_state=42)
        result = m.fit_transform(data)
        assert result.shape[0] == data.shape[0]

    def test_lens_pca1(self):
        """PCA1 lens produces valid output."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, lens="pca1", random_state=42)
        result = m.fit_transform(data)
        assert result.shape[0] == data.shape[0]

    def test_lens_diffusion1(self):
        """Diffusion1 lens produces valid output."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll(n=100)).float()
        m = ReebGraphModule(n_bins=5, lens="diffusion1", lens_k=10, lens_t=1, random_state=42)
        result = m.fit_transform(data)
        assert result.shape[0] == data.shape[0]

    def test_deterministic(self):
        """Same random_state produces same membership matrix."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m1 = ReebGraphModule(n_bins=5, random_state=42)
        m2 = ReebGraphModule(n_bins=5, random_state=42)
        r1 = m1.fit_transform(data)
        r2 = m2.fit_transform(data)
        torch.testing.assert_close(r1, r2)

    def test_invalid_lens_raises(self):
        """Invalid lens name raises ValueError."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(lens="nonexistent")
        with pytest.raises(ValueError, match="Unknown lens"):
            m.fit(data)

    def test_not_fitted_raises(self):
        """transform() and adjacency_matrix() raise before fit."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        m = ReebGraphModule()
        data = torch.randn(50, 3)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.transform(data)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.adjacency_matrix()
