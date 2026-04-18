"""Tests for density_spike data transform."""
import numpy as np
import pytest

from manylatents.data.transforms import density_spike


def _make_blob(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class TestDensitySpike:
    def test_output_size(self):
        data = _make_blob(200, 3, seed=0)
        # Use center_idx so radius is in original-data units
        augmented, label = density_spike(
            data, center_idx=0, radius=1.0, multiplier=3, noise=0.01
        )
        # Expect: original n + (n_in_region * (multiplier - 1)) duplicates.
        n_region = int(np.sum(np.linalg.norm(data - data[0], axis=1) <= 1.0))
        expected = 200 + n_region * 2
        assert augmented.shape == (expected, 3)
        assert label.shape == (expected,)
        assert label.dtype == np.int8

    def test_label_partition(self):
        data = _make_blob(100, 3, seed=1)
        augmented, label = density_spike(
            data, center_idx=5, radius=0.8, multiplier=4, noise=0.01
        )
        # First n entries are originals (label 0), rest are spikes (label 1).
        assert (label[:100] == 0).all()
        assert (label[100:] == 1).all()

    def test_deterministic(self):
        data = _make_blob(150, 3, seed=2)
        a1, l1 = density_spike(
            data, center_idx=10, radius=0.5, multiplier=5, noise=0.05, random_state=7
        )
        a2, l2 = density_spike(
            data, center_idx=10, radius=0.5, multiplier=5, noise=0.05, random_state=7
        )
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(l1, l2)

    def test_seed_changes_output(self):
        data = _make_blob(150, 3, seed=3)
        a1, _ = density_spike(
            data, center_idx=0, radius=0.7, multiplier=3, noise=0.05, random_state=1
        )
        a2, _ = density_spike(
            data, center_idx=0, radius=0.7, multiplier=3, noise=0.05, random_state=2
        )
        # Originals identical, spike portion differs because of noise RNG.
        n_orig = 150
        np.testing.assert_array_equal(a1[:n_orig], a2[:n_orig])
        assert not np.allclose(a1[n_orig:], a2[n_orig:])

    def test_zero_noise_exact_duplicates(self):
        data = _make_blob(80, 3, seed=4)
        augmented, label = density_spike(
            data, center_idx=0, radius=1e9, multiplier=2, noise=0.0
        )
        # Every original gets one exact duplicate.
        assert augmented.shape[0] == 160
        assert (label[:80] == 0).all()
        assert (label[80:] == 1).all()
        np.testing.assert_array_equal(augmented[:80], augmented[80:])

    def test_multiplier_one_is_noop_for_data(self):
        data = _make_blob(60, 3, seed=5)
        augmented, label = density_spike(
            data, center_idx=0, radius=10.0, multiplier=1, noise=0.05
        )
        assert augmented.shape == data.shape
        assert (label == 0).all()
        np.testing.assert_array_equal(augmented, data)

    def test_no_points_in_region(self):
        data = _make_blob(100, 3, seed=6)
        # Center far from any point, tiny radius
        far_center = np.full(3, 1e6, dtype=np.float32)
        augmented, label = density_spike(
            data, center=far_center, radius=1e-6, multiplier=5, noise=0.01
        )
        assert augmented.shape == data.shape
        assert (label == 0).all()

    def test_center_and_center_idx_mutually_exclusive(self):
        data = _make_blob(50, 3, seed=7)
        with pytest.raises(ValueError):
            density_spike(data)
        with pytest.raises(ValueError):
            density_spike(data, center=np.zeros(3), center_idx=0)

    def test_multiplier_must_be_positive(self):
        data = _make_blob(50, 3, seed=8)
        with pytest.raises(ValueError):
            density_spike(data, center_idx=0, multiplier=0)

    def test_center_shape_validation(self):
        data = _make_blob(50, 3, seed=9)
        with pytest.raises(ValueError):
            density_spike(data, center=np.zeros(2), radius=0.5)
