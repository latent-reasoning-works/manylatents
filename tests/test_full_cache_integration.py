"""Integration test: full evaluate_embeddings with cache."""
import sys
import types
import numpy as np
import pytest
from omegaconf import OmegaConf

# Mock dogma namespace if not installed
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


def test_extract_k_requirements_and_prewarm():
    """End-to-end: sleuther extracts requirements, prewarm populates cache."""
    from manylatents.experiment import extract_k_requirements, prewarm_cache

    cfgs = {
        "embedding.knn_preservation": OmegaConf.create({
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 10,
        }),
        "embedding.lid": OmegaConf.create({
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 20,
        }),
    }

    reqs = extract_k_requirements(cfgs)
    assert reqs["emb_k"] == {10, 20}

    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)

    class FakeDS:
        data = rng.randn(30, 5).astype(np.float32)

    cache = prewarm_cache(cfgs, emb, FakeDS())
    # Should have pre-warmed with max_k=20
    cached_k, _, _ = cache[id(emb)]
    assert cached_k == 20


def test_full_pipeline_with_cache():
    """End-to-end: sleuther + metrics all sharing one cache."""
    from manylatents.experiment import extract_k_requirements, prewarm_cache
    from manylatents.metrics.knn_preservation import KNNPreservation
    from manylatents.metrics.lid import LocalIntrinsicDimensionality
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    from manylatents.utils.metrics import compute_eigenvalues

    rng = np.random.RandomState(42)
    emb = rng.randn(50, 2).astype(np.float32)
    high_dim = rng.randn(50, 10).astype(np.float32)
    A = rng.randn(50, 50).astype(np.float64)
    A = A @ A.T  # symmetric PSD

    class FakeDS:
        data = high_dim

    class FakeModule:
        def affinity_matrix(self, use_symmetric=False):
            return A

    cfgs = {
        "embedding.knn_preservation": OmegaConf.create({
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 10,
        }),
        "embedding.lid": OmegaConf.create({
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 5,
        }),
        "module.spectral_gap_ratio": OmegaConf.create({
            "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
        }),
    }

    # Sleuther extracts requirements
    reqs = extract_k_requirements(cfgs)
    assert reqs["emb_k"] == {10, 5}
    assert 10 in reqs["data_k"]
    assert reqs["spectral"] is True

    # Prewarm populates cache
    module = FakeModule()
    cache = prewarm_cache(cfgs, emb, FakeDS(), module=module)
    assert id(emb) in cache
    assert id(high_dim) in cache
    assert "eigenvalues" in cache

    # Metrics use shared cache — no recomputation
    knn_result = KNNPreservation(emb, FakeDS(), n_neighbors=10, cache=cache)
    assert isinstance(knn_result, float)

    lid_result = LocalIntrinsicDimensionality(emb, k=5, cache=cache)
    assert isinstance(lid_result, float)

    gap_result = SpectralGapRatio(emb, module=module, cache=cache)
    assert isinstance(gap_result, float)
    assert not np.isnan(gap_result)
