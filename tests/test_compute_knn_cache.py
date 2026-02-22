"""Tests for cache-aware compute_knn."""
import numpy as np
import pytest
from manylatents.utils.metrics import _content_key, compute_knn


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
    assert _content_key(sample_data) in cache
    cached_k, _, _ = cache[_content_key(sample_data)]
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
    cached_k, _, _ = cache[_content_key(sample_data)]
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


def test_compute_knn_k_exceeds_n_samples():
    """k >= n_samples should clamp and warn."""
    small_data = np.random.RandomState(42).randn(10, 3).astype(np.float32)
    with pytest.warns(UserWarning, match="Clamping k"):
        dists, idxs = compute_knn(small_data, k=15)
    assert dists.shape[1] <= 10
    assert idxs.shape[1] <= 10


def test_compute_knn_k_equals_n_samples():
    """k == n_samples should also clamp."""
    data = np.random.RandomState(42).randn(5, 2).astype(np.float32)
    with pytest.warns(UserWarning, match="Clamping k"):
        dists, idxs = compute_knn(data, k=5)
    assert dists.shape[0] == 5


def test_compute_knn_content_key_matches(sample_data):
    """A copy of the array shares the same cache entry."""
    cache = {}
    compute_knn(sample_data, k=10, cache=cache)
    data_copy = sample_data.copy()
    dists, idxs = compute_knn(data_copy, k=5, cache=cache)
    # Should reuse cached result (k=10 >= k=5), not recompute
    assert len(cache) == 1
    assert dists.shape == (50, 6)  # k+1 with self


def test_disk_cache_roundtrip(sample_data, tmp_path):
    """Disk cache saves and loads dataset kNN correctly."""
    from types import SimpleNamespace
    from manylatents.experiment import prewarm_cache

    metric_cfgs = {
        "embedding.trustworthiness": SimpleNamespace(
            _target_="manylatents.metrics.trustworthiness.Trustworthiness",
            n_neighbors=10,
        ),
    }
    dataset = SimpleNamespace(data=sample_data)

    # First call: computes and saves to disk
    cache1 = prewarm_cache(metric_cfgs, sample_data, dataset, knn_cache_dir=str(tmp_path))
    npz_files = list((tmp_path / "knn").glob("*.npz"))
    assert len(npz_files) == 1

    # Second call: loads from disk
    cache2 = prewarm_cache(metric_cfgs, sample_data, dataset, knn_cache_dir=str(tmp_path))
    key = _content_key(sample_data)
    np.testing.assert_array_equal(cache1[key][1], cache2[key][1])
    np.testing.assert_array_equal(cache1[key][2], cache2[key][2])
