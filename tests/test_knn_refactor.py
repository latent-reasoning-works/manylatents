"""Verify kNN refactor: both import paths work, same function object."""
import numpy as np


def test_import_from_new_path():
    from manylatents.utils.knn import compute_knn
    assert callable(compute_knn)


def test_import_from_old_path():
    from manylatents.utils.metrics import compute_knn
    assert callable(compute_knn)


def test_both_paths_same_function():
    from manylatents.utils.knn import compute_knn as knn_new
    from manylatents.utils.metrics import compute_knn as knn_old
    assert knn_new is knn_old


def test_content_key_same_function():
    from manylatents.utils.knn import _content_key as key_new
    from manylatents.utils.metrics import _content_key as key_old
    assert key_new is key_old


def test_compute_knn_basic():
    from manylatents.utils.knn import compute_knn
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5)).astype(np.float32)
    dists, idxs = compute_knn(X, k=5, include_self=False)
    assert dists.shape == (50, 5)
    assert idxs.shape == (50, 5)
