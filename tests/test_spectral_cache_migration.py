"""Tests that spectral metrics work with cache-aware compute_eigenvalues."""
import numpy as np
import pytest


class FakeModule:
    def __init__(self, matrix):
        self._matrix = matrix

    def affinity_matrix(self, use_symmetric=False):
        return self._matrix


@pytest.fixture
def module_and_embeddings():
    rng = np.random.RandomState(42)
    A = rng.randn(20, 20)
    A = A @ A.T  # symmetric PSD
    emb = rng.randn(20, 2).astype(np.float32)
    return FakeModule(A), emb


def test_spectral_gap_ratio_with_cache(module_and_embeddings):
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    mod, emb = module_and_embeddings
    cache = {}
    result = SpectralGapRatio(emb, module=mod, cache=cache)
    assert isinstance(result, float)
    assert result > 0
    assert "eigenvalues" in cache


def test_spectral_decay_rate_with_cache(module_and_embeddings):
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate
    mod, emb = module_and_embeddings
    cache = {}
    result = SpectralDecayRate(emb, module=mod, cache=cache, top_k=10)
    assert isinstance(result, float)
    assert "eigenvalues" in cache


def test_spectral_metrics_share_cache(module_and_embeddings):
    """Two spectral metrics sharing same cache only compute eigenvalues once."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate
    mod, emb = module_and_embeddings

    cache = {}
    r1 = SpectralGapRatio(emb, module=mod, cache=cache)
    # Cache should now have eigenvalues
    assert "eigenvalues" in cache
    eigs_first = cache["eigenvalues"].copy()

    r2 = SpectralDecayRate(emb, module=mod, cache=cache, top_k=10)
    # Eigenvalues should be identical (reused, not recomputed)
    np.testing.assert_array_equal(cache["eigenvalues"], eigs_first)

    assert isinstance(r1, float) and r1 > 0
    assert isinstance(r2, float)


def test_spectral_gap_ratio_from_prewarmed_cache():
    """SpectralGapRatio can use a pre-warmed cache (no module needed)."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    cache = {"eigenvalues": np.array([10.0, 2.0, 1.0])}
    result = SpectralGapRatio(embeddings=np.zeros((3, 2)), cache=cache)
    np.testing.assert_allclose(result, 5.0)


def test_affinity_spectrum_with_cache(module_and_embeddings):
    from manylatents.metrics.affinity_spectrum import AffinitySpectrum
    mod, emb = module_and_embeddings
    cache = {}
    result = AffinitySpectrum(dataset=None, embeddings=emb, module=mod, top_k=5, cache=cache)
    assert len(result) == 5
    assert "eigenvalues" in cache
