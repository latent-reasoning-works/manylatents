"""Tests for eigenvalue cache computation and sharing."""
import sys
import types
from unittest import mock

import numpy as np
import pytest

# Mock the manylatents.dogma namespace extension if not installed,
# so that importing manylatents.experiment doesn't fail.
if "manylatents.dogma" not in sys.modules:
    _dogma = types.ModuleType("manylatents.dogma")
    _encoders = types.ModuleType("manylatents.dogma.encoders")
    _base = types.ModuleType("manylatents.dogma.encoders.base")
    _base.FoundationEncoder = type("FoundationEncoder", (), {})
    _dogma.encoders = _encoders
    _encoders.base = _base
    sys.modules["manylatents.dogma"] = _dogma
    sys.modules["manylatents.dogma.encoders"] = _encoders
    sys.modules["manylatents.dogma.encoders.base"] = _base

from manylatents.experiment import _compute_eigenvalue_cache


def test_compute_eigenvalue_cache_symmetric():
    """Eigenvalue cache computes sorted eigenvalues from symmetric matrix."""
    # Create a simple symmetric PSD matrix
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={None})
    assert (True, None) in cache
    eigs = cache[(True, None)]
    # Eigenvalues of [[2,1],[1,2]] are 1 and 3
    assert len(eigs) == 2
    # Should be sorted descending
    assert eigs[0] >= eigs[1]
    np.testing.assert_allclose(eigs, [3.0, 1.0], atol=1e-10)


def test_compute_eigenvalue_cache_top_k():
    """Eigenvalue cache respects top_k parameter."""
    A = np.eye(5) * np.arange(1, 6)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={3})
    eigs = cache[(True, 3)]
    assert len(eigs) == 3


def test_eigenvalue_cache_shared_across_metrics():
    """Two metrics requesting different top_k get correct cache entries."""
    A = np.random.RandomState(42).randn(10, 10)
    A = A @ A.T  # Make symmetric PSD

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={None, 5})
    assert (True, None) in cache
    assert (True, 5) in cache
    # Full spectrum should contain top-5
    np.testing.assert_allclose(cache[(True, None)][:5], cache[(True, 5)])


def test_compute_eigenvalue_cache_no_affinity():
    """Eigenvalue cache returns empty dict if module has no affinity_matrix."""

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            raise NotImplementedError("No affinity matrix")

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={None})
    assert cache == {}
