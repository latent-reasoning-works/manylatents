import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from manylatents.metrics.spectral_smoothness import SpectralSmoothness
from manylatents.utils.knn import compute_knn


def _plane(n=800, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    return (np.c_[rng.uniform(-1, 1, size=(n, 2)), np.zeros((n, 18))] @ Q).astype(np.float32)


def _direct(X, j, k=15):
    _, idx = compute_knn(X, k=k, include_self=False)
    n = len(X)
    A = sp.csr_matrix((np.ones(idx.size), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    d = np.asarray(A.sum(1)).ravel()
    lam = np.sort(eigsh(sp.diags(d ** -0.5) @ A @ sp.diags(d ** -0.5), k=j, which="LA", return_eigenvectors=False))[::-1]
    return 1 - lam[j - 1]


def test_default_is_eigen_index_3():
    X = _plane(seed=5)
    assert SpectralSmoothness(X) == SpectralSmoothness(X, eigen_index=3)


def test_smooth_plane_is_small_and_high_dim_gaussian_is_large():
    flat = SpectralSmoothness(_plane())
    rough = SpectralSmoothness(np.random.default_rng(1).normal(size=(800, 20)).astype(np.float32))
    assert 0.0 <= flat < 0.05
    assert rough > 5 * flat


@pytest.mark.parametrize("j", [2, 3, 5, 8])
def test_matches_direct_eigensolve(j):
    X = _plane(n=400, seed=4)
    assert abs(SpectralSmoothness(X, eigen_index=j) - _direct(X, j)) < 1e-6


def test_nondecreasing_in_eigen_index():
    X = _plane(n=500, seed=6)
    v = [SpectralSmoothness(X, eigen_index=j) for j in (2, 3, 4, 6, 10)]
    assert all(a <= b + 1e-12 for a, b in zip(v, v[1:]))


def test_deterministic():
    X = _plane(seed=3)
    assert SpectralSmoothness(X, eigen_index=4) == SpectralSmoothness(X, eigen_index=4)


@pytest.mark.parametrize("bad", [1, 0, -1, 2.5, True])
def test_rejects_invalid_eigen_index(bad):
    with pytest.raises(ValueError):
        SpectralSmoothness(_plane(n=100), eigen_index=bad)


def test_registered_and_listed():
    from manylatents.metrics import list_metrics
    assert "SpectralSmoothness" in list_metrics()
