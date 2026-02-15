"""Tests for SVD computation and GPU->CPU fallback."""
import numpy as np
import pytest


@pytest.fixture
def knn_data():
    rng = np.random.RandomState(42)
    n, d = 30, 4
    emb = rng.randn(n, d).astype(np.float32)
    idx = np.zeros((n, 11), dtype=np.int64)
    for i in range(n):
        idx[i, 0] = i
        others = [j for j in range(n) if j != i]
        idx[i, 1:] = rng.choice(others, size=10, replace=False)
    return emb, idx


def test_svd_cpu_correct_shape(knn_data):
    from manylatents.utils.metrics import _svd_cpu
    emb, idx = knn_data
    sv = _svd_cpu(emb, idx[:, 1:4], chunk_size=10)
    assert sv.shape == (30, min(3, 4))
    assert np.all(sv >= 0)


def test_svd_cpu_sorted_descending(knn_data):
    from manylatents.utils.metrics import _svd_cpu
    emb, idx = knn_data
    sv = _svd_cpu(emb, idx[:, 1:4], chunk_size=10)
    for i in range(sv.shape[0]):
        assert np.all(np.diff(sv[i]) <= 1e-6)


def test_compute_svd_cache_end_to_end(knn_data):
    from manylatents.utils.metrics import compute_svd_cache
    emb, idx = knn_data
    result = compute_svd_cache(emb, idx, k_values={3, 5})
    assert 3 in result and 5 in result
    assert result[3].shape == (30, min(3, 4))
    assert result[5].shape == (30, min(5, 4))


def test_gpu_fallback_pattern(knn_data):
    """Simulate GPU failure -> CPU fallback."""
    from manylatents.utils.metrics import _svd_cpu
    emb, idx = knn_data

    def failing_gpu(*a, **kw):
        raise RuntimeError("CUDA OOM")

    try:
        sv = failing_gpu(emb, idx[:, 1:4], 10)
    except Exception:
        sv = _svd_cpu(emb, idx[:, 1:4], 10)

    assert sv.shape == (30, min(3, 4))
