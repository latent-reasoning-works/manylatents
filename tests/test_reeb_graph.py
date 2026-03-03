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
        assert result.shape[1] == m.kernel_matrix().shape[0]  # M matches adjacency

    def test_hard_assignment(self):
        """Each point belongs to at most one Reeb node (hard assignment)."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        # Even with overlap, assignment is hard — overlap only helps edge detection
        m = ReebGraphModule(n_bins=5, overlap=0.5, random_state=42)
        result = m.fit_transform(data)
        memberships_per_point = result.sum(dim=1)
        assert (memberships_per_point <= 1).all()

    def test_overlap_improves_connectivity(self):
        """Overlap>0 can produce more edges than overlap=0."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m_strict = ReebGraphModule(n_bins=8, overlap=0.0, random_state=42)
        m_overlap = ReebGraphModule(n_bins=8, overlap=0.5, random_state=42)
        m_strict.fit(data)
        m_overlap.fit(data)
        # Overlap should produce at least as many edges (more connectivity)
        assert m_overlap.structural_summary["n_edges"] >= m_strict.structural_summary["n_edges"]

    def test_membership_is_binary(self):
        """Membership matrix values are 0 or 1."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, overlap=0.25, random_state=42)
        result = m.fit_transform(data)
        unique_vals = set(torch.unique(result).numpy().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_kernel_matrix(self):
        """kernel_matrix() returns a valid (M, M) symmetric adjacency."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        K = m.kernel_matrix()
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
        """node_coordinates has shape (M, D) matching kernel_matrix size."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        data = torch.from_numpy(_make_swissroll()).float()
        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        K = m.kernel_matrix()
        coords = m.node_coordinates
        assert coords is not None
        assert coords.shape[0] == K.shape[0]  # M nodes
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
        """transform() and kernel_matrix() raise before fit."""
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        m = ReebGraphModule()
        data = torch.randn(50, 3)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.transform(data)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.kernel_matrix()
