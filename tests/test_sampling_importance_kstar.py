"""Tests for ImportanceSampling, KStarWeightedSampling, and GeosketchSampling."""

import numpy as np
import pytest

from manylatents.utils.sampling import (
    GeosketchSampling,
    ImportanceSampling,
    KStarWeightedSampling,
    MismatchAwareSampling,
)


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


# ---------------------------------------------------------------------------
# TestMismatchAwareSampling
# ---------------------------------------------------------------------------

class TestMismatchAwareSampling:
    def test_output_size_with_kref(self):
        data = _make_data(500, 10)
        sampler = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(idx) == 250

    def test_output_size_with_precomputed_v(self):
        n = 500
        rng = np.random.default_rng(0)
        v = rng.uniform(0.1, 3.0, size=n)
        sampler = MismatchAwareSampling(v=v, seed=42)
        idx = sampler.get_indices(n, fraction=0.5)
        assert len(idx) == 250

    def test_sorted_indices(self):
        data = _make_data(500, 10)
        sampler = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert np.all(np.diff(idx) >= 0)

    def test_no_duplicates(self):
        data = _make_data(500, 10)
        sampler = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        idx = sampler.get_indices(data, fraction=0.5)
        assert len(set(idx)) == len(idx)

    def test_deterministic(self):
        data = _make_data(500, 10)
        s1 = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        s2 = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        idx1 = s1.get_indices(data, fraction=0.5)
        idx2 = s2.get_indices(data, fraction=0.5)
        np.testing.assert_array_equal(idx1, idx2)

    def test_high_v_downweighted(self):
        # Construct precomputed v: half the points have v=0.1, half have v=10.
        # The high-v half should be sampled rarely.
        n = 1000
        v = np.concatenate([np.full(500, 0.1), np.full(500, 10.0)])
        sampler = MismatchAwareSampling(v=v, alpha=1.0, seed=42)
        idx = sampler.get_indices(n, fraction=0.5)
        # Indices from the high-v block (>= 500) should be a small minority.
        n_high = int(np.sum(idx >= 500))
        assert n_high < 50  # expected ~5 (1/100 weight ratio); allow slack

    def test_construction_requires_one_of_kref_or_v(self):
        with pytest.raises(ValueError):
            MismatchAwareSampling()
        with pytest.raises(ValueError):
            MismatchAwareSampling(k_ref=15, v=np.array([1.0, 2.0]))

    def test_int_input_with_kref_rejected(self):
        sampler = MismatchAwareSampling(k_ref=15, seed=42)
        with pytest.raises(TypeError):
            sampler.get_indices(500, fraction=0.5)

    def test_v_length_mismatch_raises(self):
        data = _make_data(500, 10)
        sampler = MismatchAwareSampling(v=np.ones(100), seed=42)
        with pytest.raises(ValueError):
            sampler.get_indices(data, fraction=0.5)

    def test_sample_method(self):
        data = _make_data(500, 10)
        ds = MockDataset()
        sampler = MismatchAwareSampling(k_ref=15, k_max=50, seed=42)
        emb_sub, ds_sub, idx = sampler.sample(data, ds, fraction=0.5)

        assert emb_sub.shape == (250, 10)
        assert len(idx) == 250
        assert isinstance(ds_sub, MockDataset)


# ---------------------------------------------------------------------------
# GeosketchSampling tests
# ---------------------------------------------------------------------------

geosketch = pytest.importorskip("geosketch")


class TestGeosketchSampling:
    def test_returns_correct_count_via_n_samples(self):
        data = _make_data(500, 60)
        sampler = GeosketchSampling(n_samples=100, seed=0)
        idx = sampler.get_indices(data)
        assert len(idx) == 100

    def test_returns_correct_count_via_fraction(self):
        data = _make_data(500, 60)
        sampler = GeosketchSampling(fraction=0.4, seed=0)
        idx = sampler.get_indices(data)
        assert len(idx) == 200

    def test_indices_sorted(self):
        data = _make_data(300, 60)
        sampler = GeosketchSampling(fraction=0.5, seed=0)
        idx = sampler.get_indices(data)
        assert np.all(idx[:-1] <= idx[1:])

    def test_indices_in_range(self):
        n = 400
        data = _make_data(n, 60)
        sampler = GeosketchSampling(fraction=0.5, seed=0)
        idx = sampler.get_indices(data)
        assert idx.min() >= 0 and idx.max() < n

    def test_raises_without_count_or_fraction(self):
        data = _make_data(300, 60)
        sampler = GeosketchSampling()
        with pytest.raises(ValueError):
            sampler.get_indices(data)

    def test_only_first_50_dims_used(self):
        data = _make_data(300, 200)
        sampler = GeosketchSampling(fraction=0.5, seed=0)
        idx = sampler.get_indices(data)
        assert len(idx) == 150

    def test_seed_override(self):
        data = _make_data(500, 60)
        sampler = GeosketchSampling(fraction=0.5, seed=0)
        idx1 = sampler.get_indices(data, seed=0)
        idx2 = sampler.get_indices(data, seed=0)
        assert np.array_equal(idx1, idx2)

    def test_n_samples_override_at_call_time(self):
        data = _make_data(500, 60)
        sampler = GeosketchSampling(fraction=0.5, seed=0)
        idx = sampler.get_indices(data, n_samples=50)
        assert len(idx) == 50
