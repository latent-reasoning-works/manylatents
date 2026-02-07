"""Tests for SVD cache shared between ParticipationRatio and TangentSpaceApproximation."""

import numpy as np
import pytest

from manylatents.metrics.participation_ratio import ParticipationRatio
from manylatents.metrics.tangent_space import TangentSpaceApproximation
from manylatents.utils.metrics import compute_knn, compute_svd_cache


@pytest.fixture
def synthetic_data():
    """Generate reproducible synthetic data for testing."""
    rng = np.random.RandomState(42)
    n, d = 200, 10
    return rng.randn(n, d).astype(np.float32)


@pytest.fixture
def knn_cache(synthetic_data):
    """Precompute kNN cache."""
    return compute_knn(synthetic_data, k=50, include_self=True)


class TestComputeSvdCache:
    def test_basic_shape(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        cache = compute_svd_cache(synthetic_data, indices, {10, 25})
        assert set(cache.keys()) == {10, 25}
        assert cache[10].shape == (200, 10)   # min(k=10, d=10)
        assert cache[25].shape == (200, 10)   # min(k=25, d=10)

    def test_singular_values_nonnegative(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        cache = compute_svd_cache(synthetic_data, indices, {15})
        assert np.all(cache[15] >= 0)

    def test_empty_k_values(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        cache = compute_svd_cache(synthetic_data, indices, set())
        assert cache == {}


class TestParticipationRatioWithCache:
    def test_cached_matches_uncached(self, synthetic_data, knn_cache):
        """PR with SVD cache must match PR without it."""
        _, indices = knn_cache
        k = 20
        svd_cache = compute_svd_cache(synthetic_data, indices, {k})

        pr_uncached = ParticipationRatio(
            synthetic_data, n_neighbors=k, _knn_cache=knn_cache,
        )
        pr_cached = ParticipationRatio(
            synthetic_data, n_neighbors=k, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        np.testing.assert_allclose(pr_cached, pr_uncached, rtol=1e-10)

    def test_per_sample_cached_matches_uncached(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        k = 15
        svd_cache = compute_svd_cache(synthetic_data, indices, {k})

        pr_uncached = ParticipationRatio(
            synthetic_data, n_neighbors=k, return_per_sample=True,
            _knn_cache=knn_cache,
        )
        pr_cached = ParticipationRatio(
            synthetic_data, n_neighbors=k, return_per_sample=True,
            _knn_cache=knn_cache, _svd_cache=svd_cache,
        )
        np.testing.assert_allclose(pr_cached, pr_uncached, rtol=1e-10)

    def test_missing_k_falls_back(self, synthetic_data, knn_cache):
        """If SVD cache doesn't contain needed k, metric computes inline."""
        _, indices = knn_cache
        svd_cache = compute_svd_cache(synthetic_data, indices, {10})

        # Request k=20 which is NOT in the cache
        pr_result = ParticipationRatio(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        pr_expected = ParticipationRatio(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
        )
        np.testing.assert_allclose(pr_result, pr_expected, rtol=1e-10)

    def test_none_svd_cache_works(self, synthetic_data, knn_cache):
        """Passing _svd_cache=None should work identically to not passing it."""
        pr_a = ParticipationRatio(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
        )
        pr_b = ParticipationRatio(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
            _svd_cache=None,
        )
        np.testing.assert_allclose(pr_a, pr_b, rtol=1e-10)


class TestTangentSpaceWithCache:
    def test_cached_matches_uncached(self, synthetic_data, knn_cache):
        """TSA with SVD cache must match TSA without it."""
        _, indices = knn_cache
        k = 20
        svd_cache = compute_svd_cache(synthetic_data, indices, {k})

        tsa_uncached = TangentSpaceApproximation(
            synthetic_data, n_neighbors=k, _knn_cache=knn_cache,
        )
        tsa_cached = TangentSpaceApproximation(
            synthetic_data, n_neighbors=k, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        assert tsa_cached == tsa_uncached

    def test_per_sample_cached_matches_uncached(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        k = 15
        svd_cache = compute_svd_cache(synthetic_data, indices, {k})

        tsa_uncached, _ = TangentSpaceApproximation(
            synthetic_data, n_neighbors=k, return_per_sample=True,
            _knn_cache=knn_cache,
        )
        tsa_cached, _ = TangentSpaceApproximation(
            synthetic_data, n_neighbors=k, return_per_sample=True,
            _knn_cache=knn_cache, _svd_cache=svd_cache,
        )
        np.testing.assert_array_equal(tsa_cached, tsa_uncached)

    def test_missing_k_falls_back(self, synthetic_data, knn_cache):
        _, indices = knn_cache
        svd_cache = compute_svd_cache(synthetic_data, indices, {10})

        tsa_result = TangentSpaceApproximation(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        tsa_expected = TangentSpaceApproximation(
            synthetic_data, n_neighbors=20, _knn_cache=knn_cache,
        )
        assert tsa_result == tsa_expected


class TestMultipleKValues:
    def test_pr_and_tsa_different_k_shared_cache(self, synthetic_data, knn_cache):
        """PR at k=10 and TSA at k=25 should both work from a single SVD cache."""
        _, indices = knn_cache
        svd_cache = compute_svd_cache(synthetic_data, indices, {10, 25})

        pr_cached = ParticipationRatio(
            synthetic_data, n_neighbors=10, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        pr_expected = ParticipationRatio(
            synthetic_data, n_neighbors=10, _knn_cache=knn_cache,
        )
        np.testing.assert_allclose(pr_cached, pr_expected, rtol=1e-10)

        tsa_cached = TangentSpaceApproximation(
            synthetic_data, n_neighbors=25, _knn_cache=knn_cache,
            _svd_cache=svd_cache,
        )
        tsa_expected = TangentSpaceApproximation(
            synthetic_data, n_neighbors=25, _knn_cache=knn_cache,
        )
        assert tsa_cached == tsa_expected
