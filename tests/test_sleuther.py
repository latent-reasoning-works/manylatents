"""Tests for config sleuther functions (extract_k_requirements, prewarm_cache)."""
import sys
import types
import numpy as np
import pytest
from omegaconf import OmegaConf
from manylatents.utils.metrics import _content_key

# Mock the manylatents.dogma namespace extension if not installed
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


def _make_metric_cfgs(metrics_dict):
    """Helper to create a flattened metric config dict."""
    cfgs = {}
    for name, params in metrics_dict.items():
        cfgs[name] = OmegaConf.create(params)
    return cfgs


def test_extract_k_requirements_embedding_metrics():
    from manylatents.evaluate import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "trustworthiness": {
            "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
            "n_neighbors": 25,
            "at": "embedding",
        },
        "lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 10,
            "at": "embedding",
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert 25 in reqs["knn"]["embedding"]
    assert 10 in reqs["knn"]["embedding"]


def test_extract_k_requirements_data_metrics():
    from manylatents.evaluate import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "knn_preservation": {
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 15,
            "at": "dataset",
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert 15 in reqs["knn"]["dataset"]


def test_extract_k_requirements_spectral():
    from manylatents.evaluate import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "spectral_gap_ratio": {
            "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
            "at": "module",
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert reqs["spectral"] is True


def test_extract_k_requirements_empty():
    from manylatents.evaluate import extract_k_requirements
    reqs = extract_k_requirements({})
    assert reqs["knn"] == {}
    assert reqs["spectral"] is False


def test_prewarm_cache_populates():
    from manylatents.evaluate import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)
    data = rng.randn(30, 10).astype(np.float32)

    class FakeDataset:
        pass
    ds = FakeDataset()
    ds.data = data

    cfgs = _make_metric_cfgs({
        "knn_preservation": {
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 10,
            "at": "embedding",
        },
    })
    cache = prewarm_cache(cfgs, emb, ds)
    assert _content_key(emb) in cache


def test_prewarm_cache_uses_max_k():
    from manylatents.evaluate import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)

    class FakeDataset:
        data = rng.randn(30, 5).astype(np.float32)

    cfgs = _make_metric_cfgs({
        "lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 5,
            "at": "embedding",
        },
        "lid_large": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 20,
            "at": "embedding",
        },
    })
    cache = prewarm_cache(cfgs, emb, FakeDataset())
    cached_k, _, _ = cache[_content_key(emb)]
    assert cached_k == 20


def test_prewarm_cache_spectral():
    from manylatents.evaluate import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(10, 2).astype(np.float32)
    A = rng.randn(10, 10)
    A = A @ A.T

    class FakeModule:
        def affinity(self, use_symmetric=False):
            return A

    class FakeDataset:
        data = rng.randn(10, 5).astype(np.float32)

    cfgs = _make_metric_cfgs({
        "spectral_gap_ratio": {
            "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
            "at": "module",
        },
    })
    cache = prewarm_cache(cfgs, emb, FakeDataset(), module=FakeModule())
    assert "eigenvalues" in cache


def test_prewarm_cache_no_data_attribute():
    """prewarm_cache should not crash if dataset lacks .data."""
    from manylatents.evaluate import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(20, 2).astype(np.float32)

    class NoDataDataset:
        pass

    cfgs = _make_metric_cfgs({
        "lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 5,
            "at": "embedding",
        },
    })
    cache = prewarm_cache(cfgs, emb, NoDataDataset())
    assert _content_key(emb) in cache


def test_extract_k_requirements_from_names():
    """extract_k_requirements accepts list[str] of metric names."""
    from manylatents.evaluate import extract_k_requirements
    reqs = extract_k_requirements(["Trustworthiness", "LocalIntrinsicDimensionality"])
    assert len(reqs["knn"].get("embedding", set())) > 0


def test_extract_k_requirements_from_names_spectral():
    """Registry path cannot detect spectral needs (no on field).
    Spectral prewarming for programmatic API is handled by evaluate_metrics()
    which always passes module when available."""
    from manylatents.evaluate import extract_k_requirements
    reqs = extract_k_requirements(["SpectralGapRatio"])
    # Registry path has no on field — spectral detection is config-driven
    assert reqs["spectral"] is False


def test_extract_k_requirements_from_names_empty():
    from manylatents.evaluate import extract_k_requirements
    reqs = extract_k_requirements([])
    assert reqs["knn"] == {}
    assert reqs["spectral"] is False
