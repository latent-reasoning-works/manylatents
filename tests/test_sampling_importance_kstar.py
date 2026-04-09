"""Tests for ImportanceSampling and KStarWeightedSampling."""

import numpy as np
import pytest

from manylatents.utils.sampling import ImportanceSampling, KStarWeightedSampling


class MockDataset:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=500, d=10, seed=0):
    """Return (n, d) float32 data from a reproducible RNG."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def _make_dense_sparse(n_dense=400, n_sparse=100, d=10, seed=0):
    """400 dense points (std=0.1) + 100 sparse points (std=2.0)."""
    rng = np.random.default_rng(seed)
    dense = (rng.standard_normal((n_dense, d)) * 0.1).astype(np.float32)
    sparse = (rng.standard_normal((n_sparse, d)) * 2.0).astype(np.float32)
    return np.vstack([dense, sparse])


def _make_two_clusters(n_per_cluster=300, d=10, seed=0):
    """Two clusters separated by 5 in dimension 0."""
    rng = np.random.default_rng(seed)
    c0 = rng.standard_normal((n_per_cluster, d)).astype(np.float32)
    c1 = rng.standard_normal((n_per_cluster, d)).astype(np.float32)
    c1[:, 0] += 5.0
    return np.vstack([c0, c1])


# ---------------------------------------------------------------------------
# TestImportanceSampling
# ---------------------------------------------------------------------------

class TestImportanceSampling:
    def test_output_size(self):
        data = _make_data(500, 10)
        sampler = ImportanceSampling(k=10, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(idx) == 250

    def test_sorted_indices(self):
        data = _make_data(500, 10)
        sampler = ImportanceSampling(k=10, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert np.all(np.diff(idx) >= 0)

    def test_no_duplicates(self):
        data = _make_data(500, 10)
        sampler = ImportanceSampling(k=10, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(set(idx)) == len(idx)

    def test_deterministic(self):
        data = _make_data(500, 10)
        s1 = ImportanceSampling(k=10, seed=42)
        s2 = ImportanceSampling(k=10, seed=42)
        idx1 = s1.get_indices(data, fraction=0.5)
        idx2 = s2.get_indices(data, fraction=0.5)
        np.testing.assert_array_equal(idx1, idx2)

    def test_sparse_upweighted(self):
        data = _make_dense_sparse(n_dense=400, n_sparse=100, d=10, seed=0)
        sampler = ImportanceSampling(k=10, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)

        # Sparse points are indices 400..499
        sparse_in_sample = np.sum(idx >= 400)
        sparse_fraction_sample = sparse_in_sample / len(idx)
        sparse_fraction_full = 100 / 500

        # Density-inverse weighting should oversample sparse region
        assert sparse_fraction_sample > sparse_fraction_full

    def test_rejects_int_input(self):
        sampler = ImportanceSampling(k=10, seed=42)
        with pytest.raises(TypeError):
            sampler.get_indices(500, fraction=0.5)

    def test_sample_method(self):
        data = _make_data(500, 10)
        ds = MockDataset()
        sampler = ImportanceSampling(k=10, seed=42)
        emb_sub, ds_sub, idx = sampler.sample(data, ds, fraction=0.5)

        assert emb_sub.shape == (250, 10)
        assert len(idx) == 250
        assert isinstance(ds_sub, MockDataset)


# ---------------------------------------------------------------------------
# TestKStarWeightedSampling
# ---------------------------------------------------------------------------

class TestKStarWeightedSampling:
    def test_output_size(self):
        data = _make_data(500, 10)
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(idx) == 250

    def test_sorted_indices(self):
        data = _make_data(500, 10)
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert np.all(np.diff(idx) >= 0)

    def test_no_duplicates(self):
        data = _make_data(500, 10)
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(set(idx)) == len(idx)

    def test_deterministic(self):
        data = _make_data(500, 10)
        s1 = KStarWeightedSampling(k_max=50, seed=42)
        s2 = KStarWeightedSampling(k_max=50, seed=42)
        idx1 = s1.get_indices(data, fraction=0.5)
        idx2 = s2.get_indices(data, fraction=0.5)
        np.testing.assert_array_equal(idx1, idx2)

    def test_boundary_enrichment(self):
        data = _make_two_clusters(n_per_cluster=300, d=10, seed=0)
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)

        # Boundary region: points where |x_0 - 2.5| < 1.5
        sampled_x0 = data[idx, 0]
        full_x0 = data[:, 0]

        boundary_in_sample = np.sum(np.abs(sampled_x0 - 2.5) < 1.5)
        boundary_in_full = np.sum(np.abs(full_x0 - 2.5) < 1.5)

        boundary_frac_sample = boundary_in_sample / len(idx)
        boundary_frac_full = boundary_in_full / len(data)

        # k* weighting should overrepresent boundary points
        assert boundary_frac_sample > boundary_frac_full

    def test_rejects_int_input(self):
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        with pytest.raises(TypeError):
            sampler.get_indices(500, fraction=0.5)

    def test_sample_method(self):
        data = _make_data(500, 10)
        ds = MockDataset()
        sampler = KStarWeightedSampling(k_max=50, seed=42)
        emb_sub, ds_sub, idx = sampler.sample(data, ds, fraction=0.5)

        assert emb_sub.shape == (250, 10)
        assert len(idx) == 250
        assert isinstance(ds_sub, MockDataset)
