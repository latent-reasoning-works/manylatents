"""Tests for cache-aware compute_eigenvalues."""
import numpy as np
import pytest
from manylatents.utils.metrics import compute_eigenvalues


class FakeModule:
    def __init__(self, matrix):
        self._matrix = matrix
        self.call_count = 0

    def affinity_matrix(self, use_symmetric=False):
        self.call_count += 1
        return self._matrix


def test_compute_eigenvalues_no_cache():
    """Works without cache."""
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)
    mod = FakeModule(A)
    eigs = compute_eigenvalues(mod)
    np.testing.assert_allclose(eigs, [3.0, 1.0], atol=1e-10)


def test_compute_eigenvalues_populates_cache():
    """First call populates cache."""
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)
    mod = FakeModule(A)
    cache = {}
    compute_eigenvalues(mod, cache=cache)
    assert "eigenvalues" in cache


def test_compute_eigenvalues_reuses_cache():
    """Second call uses cache, doesn't call module again."""
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)
    mod = FakeModule(A)
    cache = {}
    compute_eigenvalues(mod, cache=cache)
    compute_eigenvalues(mod, cache=cache)
    assert mod.call_count == 1  # Only called once


def test_compute_eigenvalues_no_module():
    """Returns None when module is None."""
    result = compute_eigenvalues(None)
    assert result is None


def test_compute_eigenvalues_no_affinity():
    """Returns None when module raises NotImplementedError."""
    class BadModule:
        def affinity_matrix(self, use_symmetric=False):
            raise NotImplementedError
    result = compute_eigenvalues(BadModule())
    assert result is None
