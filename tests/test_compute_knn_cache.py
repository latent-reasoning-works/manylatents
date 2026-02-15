"""Tests for cache-aware compute_knn."""
import numpy as np
import pytest
from manylatents.utils.metrics import compute_knn


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    return rng.randn(50, 5).astype(np.float32)


def test_compute_knn_no_cache(sample_data):
    """compute_knn works without cache (backward compat)."""
    dists, idxs = compute_knn(sample_data, k=5)
    assert dists.shape == (50, 6)  # k+1 with self
    assert idxs.shape == (50, 6)


def test_compute_knn_cache_populates(sample_data):
    """First call populates the cache."""
    cache = {}
    compute_knn(sample_data, k=10, cache=cache)
    assert id(sample_data) in cache
    cached_k, _, _ = cache[id(sample_data)]
    assert cached_k == 10


def test_compute_knn_cache_reuses(sample_data):
    """Second call with smaller k reuses cached result."""
    cache = {}
    dists1, idxs1 = compute_knn(sample_data, k=10, cache=cache)
    dists2, idxs2 = compute_knn(sample_data, k=5, cache=cache)
    # Should be a slice of the first result
    np.testing.assert_array_equal(idxs2, idxs1[:, :6])
    np.testing.assert_array_equal(dists2, dists1[:, :6])


def test_compute_knn_cache_recomputes_larger_k(sample_data):
    """If requested k > cached k, recompute."""
    cache = {}
    compute_knn(sample_data, k=5, cache=cache)
    dists, idxs = compute_knn(sample_data, k=15, cache=cache)
    assert dists.shape == (50, 16)
    # Cache should now have the larger k
    cached_k, _, _ = cache[id(sample_data)]
    assert cached_k == 15


def test_compute_knn_cache_no_self(sample_data):
    """Cache works with include_self=False."""
    cache = {}
    compute_knn(sample_data, k=10, include_self=True, cache=cache)
    dists, idxs = compute_knn(sample_data, k=5, include_self=False, cache=cache)
    assert dists.shape == (50, 5)
    assert idxs.shape == (50, 5)


def test_compute_knn_cache_different_arrays():
    """Different arrays get separate cache entries."""
    rng = np.random.RandomState(42)
    data_a = rng.randn(30, 3).astype(np.float32)
    data_b = rng.randn(30, 3).astype(np.float32)
    cache = {}
    compute_knn(data_a, k=5, cache=cache)
    compute_knn(data_b, k=5, cache=cache)
    assert len(cache) == 2
