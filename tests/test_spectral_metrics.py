"""Tests for spectral metrics (SpectralGapRatio, SpectralDecayRate)."""
import numpy as np
import pytest


def test_spectral_gap_ratio_basic():
    """SpectralGapRatio returns lambda_1/lambda_2."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    # Eigenvalues of [[2,1],[1,2]] are 3 and 1
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    result = SpectralGapRatio(
        embeddings=np.zeros((2, 2)),
        module=FakeModule(),
    )
    np.testing.assert_allclose(result, 3.0, atol=1e-10)


def test_spectral_gap_ratio_uses_cache():
    """SpectralGapRatio uses _eigenvalue_cache when provided."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    cache = {(True, None): np.array([10.0, 2.0, 1.0])}
    result = SpectralGapRatio(
        embeddings=np.zeros((3, 2)),
        _eigenvalue_cache=cache,
    )
    np.testing.assert_allclose(result, 5.0)


def test_spectral_gap_ratio_no_module_returns_nan():
    """SpectralGapRatio returns nan when no module and no cache."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    result = SpectralGapRatio(embeddings=np.zeros((3, 2)))
    assert np.isnan(result)


def test_spectral_decay_rate_basic():
    """SpectralDecayRate fits exponential decay to eigenvalues."""
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate

    # Create eigenvalues that decay exponentially: exp(-0.5 * i)
    eigs = np.exp(-0.5 * np.arange(20))
    cache = {(True, 20): eigs}

    result = SpectralDecayRate(
        embeddings=np.zeros((20, 2)),
        _eigenvalue_cache=cache,
        top_k=20,
    )
    assert isinstance(result, float)
    assert result > 0  # Decay rate should be positive
    np.testing.assert_allclose(result, 0.5, atol=0.1)


def test_spectral_decay_rate_no_data_returns_nan():
    """SpectralDecayRate returns nan when no module and no cache."""
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate

    result = SpectralDecayRate(embeddings=np.zeros((3, 2)))
    assert np.isnan(result)
