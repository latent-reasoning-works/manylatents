"""Tests for LatentModule.extra_outputs() generic method."""
import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.latent_module_base import LatentModule


class MinimalModule(LatentModule):
    """Bare module with no matrix methods implemented."""

    def fit(self, x, y=None):
        self._is_fitted = True

    def transform(self, x):
        return x[:, :self.n_components]


class ModuleWithTrajectories(MinimalModule):
    """Module that stores Tensor trajectories."""

    def fit(self, x, y=None):
        super().fit(x, y)
        self.trajectories = torch.randn(5, x.shape[0], self.n_components)


class ModuleWithAffinity(MinimalModule):
    """Module that exposes an affinity matrix."""

    def fit(self, x, y=None):
        super().fit(x, y)
        self._n = x.shape[0]

    def affinity(self, ignore_diagonal=False, use_symmetric=False):
        return np.eye(self._n)


class TestExtraOutputsBase:
    def test_empty_for_minimal_module(self):
        m = MinimalModule()
        m.fit(torch.randn(10, 3))
        extras = m.extra_outputs()
        assert extras == {}

    def test_collects_affinity_matrix(self):
        m = ModuleWithAffinity()
        data = torch.randn(10, 3)
        m.fit(data)
        extras = m.extra_outputs()
        assert "affinity" in extras
        assert extras["affinity"].shape == (10, 10)

    def test_tensor_trajectories_converted_to_numpy(self):
        m = ModuleWithTrajectories()
        data = torch.randn(10, 3)
        m.fit(data)
        extras = m.extra_outputs()
        assert "trajectories" in extras
        assert isinstance(extras["trajectories"], np.ndarray)
        assert extras["trajectories"].shape == (5, 10, 2)

    def test_unfitted_module_returns_empty(self):
        """Unfitted modules that raise RuntimeError are silently skipped."""
        m = MinimalModule()
        # Not fitted, but base extra_outputs doesn't call methods that check _is_fitted
        # (since MinimalModule doesn't implement matrix methods at all)
        extras = m.extra_outputs()
        assert extras == {}


class TestExtraOutputsPCA:
    def test_pca_includes_matrices(self):
        from manylatents.algorithms.latent import PCAModule

        data = torch.randn(30, 5)
        m = PCAModule(n_components=2)
        m.fit(data)
        extras = m.extra_outputs()
        assert "affinity" in extras or "kernel" in extras


class TestExtraOutputsReebGraph:
    def test_reeb_includes_node_coordinates_and_summary(self):
        ripser = pytest.importorskip("ripser")  # noqa: F841
        gudhi = pytest.importorskip("gudhi")  # noqa: F841
        from manylatents.algorithms.latent.reeb_graph import ReebGraphModule

        rng = np.random.RandomState(42)
        t = 1.5 * np.pi * (1 + 2 * rng.rand(200))
        data = np.column_stack([t * np.cos(t), 30 * rng.rand(200), t * np.sin(t)])
        data = torch.from_numpy(data).float()

        m = ReebGraphModule(n_bins=5, random_state=42)
        m.fit(data)
        extras = m.extra_outputs()

        assert "adjacency" in extras
        assert "node_coordinates" in extras
        assert "structural_summary" in extras
        assert extras["node_coordinates"].shape[0] == extras["adjacency"].shape[0]
        assert isinstance(extras["structural_summary"], dict)
