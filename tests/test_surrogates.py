"""Surrogate data that keeps some structure of an array and destroys the rest."""
import numpy as np
import pytest

from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.surrogates import (
    gaussian_surrogate,
    permute_within_groups,
    random_feature_subset,
    shuffle_within_rows,
)


def test_shuffle_within_rows_keeps_each_row_multiset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 30))
    shuffled = shuffle_within_rows(x, np.random.default_rng(1))
    assert shuffled.shape == x.shape
    np.testing.assert_array_equal(np.sort(shuffled, axis=1), np.sort(x, axis=1))
    np.testing.assert_array_equal(np.abs(shuffled).max(axis=1), np.abs(x).max(axis=1))
    np.testing.assert_allclose(np.linalg.norm(shuffled, axis=1), np.linalg.norm(x, axis=1))
    assert not np.array_equal(shuffled, x)
    # rows are permuted independently, so column structure is gone
    assert not np.array_equal(np.sort(shuffled, axis=0), np.sort(x, axis=0))


def test_shuffle_is_reproducible_and_leaves_input_untouched():
    x = np.arange(12.0).reshape(3, 4)
    before = x.copy()
    a = shuffle_within_rows(x, np.random.default_rng(5))
    b = shuffle_within_rows(x, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, before)


def test_random_feature_subset():
    x = np.arange(40.0).reshape(4, 10)
    sub, idx = random_feature_subset(x, 3, np.random.default_rng(2))
    assert sub.shape == (4, 3)
    assert len(set(idx.tolist())) == 3 and np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(sub, x[:, idx])
    with pytest.raises(ValueError):
        random_feature_subset(x, 11, np.random.default_rng(2))
    with pytest.raises(ValueError):
        random_feature_subset(x, 0, np.random.default_rng(2))


def test_gaussian_surrogate_matches_mean_and_covariance():
    rng = np.random.default_rng(3)
    mixing = rng.normal(size=(5, 5))
    x = rng.normal(size=(400, 5)) @ mixing + np.array([1.0, -2.0, 0.0, 3.0, 5.0])
    surrogate = gaussian_surrogate(x, np.random.default_rng(4), n_samples=20000)
    assert surrogate.shape == (20000, 5)
    np.testing.assert_allclose(surrogate.mean(axis=0), x.mean(axis=0), atol=0.15)
    cov, target = np.cov(surrogate, rowvar=False), np.cov(x, rowvar=False)
    assert np.linalg.norm(cov - target) / np.linalg.norm(target) < 0.1


def test_gaussian_surrogate_with_more_features_than_rows():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 300))
    surrogate = gaussian_surrogate(x, np.random.default_rng(6))
    assert surrogate.shape == (20, 300)
    assert np.all(np.isfinite(surrogate))
    # the surrogate stays in the span of the centred data
    centred = x - x.mean(axis=0)
    basis = np.linalg.svd(centred, full_matrices=False)[2]
    residual = (surrogate - x.mean(axis=0)) - (surrogate - x.mean(axis=0)) @ basis.T @ basis
    assert np.abs(residual).max() < 1e-8


def test_gaussian_surrogate_needs_two_rows():
    with pytest.raises(MeasurementUnavailable):
        gaussian_surrogate(np.ones((1, 3)), np.random.default_rng(0))


def test_permute_within_groups_keeps_group_contents():
    values = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    out = permute_within_groups(values, groups, np.random.default_rng(7))
    assert out.shape == values.shape
    for g in ("a", "b"):
        assert sorted(out[groups == g].tolist()) == sorted(values[groups == g].tolist())
    np.testing.assert_array_equal(
        out, permute_within_groups(values, groups, np.random.default_rng(7))
    )
    with pytest.raises(ValueError):
        permute_within_groups(values, groups[:-1], np.random.default_rng(7))


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_gaussian_surrogate_rejects_invalid_sample_count(n_samples):
    with pytest.raises(MeasurementUnavailable):
        gaussian_surrogate(np.arange(12.).reshape(4, 3), np.random.default_rng(0), n_samples=n_samples)


def test_gaussian_surrogate_rejects_complex_data():
    with pytest.raises(MeasurementUnavailable):
        gaussian_surrogate(np.ones((3, 2)) * (1 + 2j), np.random.default_rng(0))


def test_group_permutation_refuses_missing_group_keys():
    with pytest.raises(MeasurementUnavailable):
        permute_within_groups(np.arange(4), np.array([0., 0., np.nan, np.nan]), np.random.default_rng(0))
