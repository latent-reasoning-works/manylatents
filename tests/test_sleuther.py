"""Tests for config sleuther functions."""
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
    from manylatents.experiment import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "embedding.trustworthiness": {
            "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
            "n_neighbors": 25,
        },
        "embedding.lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 10,
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert 25 in reqs["emb_k"]
    assert 10 in reqs["emb_k"]


def test_extract_k_requirements_data_metrics():
    from manylatents.experiment import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "embedding.knn_preservation": {
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 15,
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert 15 in reqs["emb_k"]
    assert 15 in reqs["data_k"]


def test_extract_k_requirements_spectral():
    from manylatents.experiment import extract_k_requirements
    cfgs = _make_metric_cfgs({
        "module.spectral_gap_ratio": {
            "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
        },
    })
    reqs = extract_k_requirements(cfgs)
    assert reqs["spectral"] is True


def test_extract_k_requirements_empty():
    from manylatents.experiment import extract_k_requirements
    reqs = extract_k_requirements({})
    assert reqs["emb_k"] == set()
    assert reqs["data_k"] == set()
    assert reqs["spectral"] is False


def test_prewarm_cache_populates():
    from manylatents.experiment import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)
    data = rng.randn(30, 10).astype(np.float32)

    class FakeDataset:
        pass
    ds = FakeDataset()
    ds.data = data

    cfgs = _make_metric_cfgs({
        "embedding.knn_preservation": {
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 10,
        },
    })
    cache = prewarm_cache(cfgs, emb, ds)
    # Should have pre-warmed both embedding and dataset kNN
    assert _content_key(emb) in cache
    assert _content_key(data) in cache


def test_prewarm_cache_uses_max_k():
    from manylatents.experiment import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)

    class FakeDataset:
        data = rng.randn(30, 5).astype(np.float32)

    cfgs = _make_metric_cfgs({
        "embedding.lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 5,
        },
        "embedding.lid_large": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "k": 20,
        },
    })
    cache = prewarm_cache(cfgs, emb, FakeDataset())
    # Should have pre-warmed with max_k=20
    cached_k, _, _ = cache[_content_key(emb)]
    assert cached_k == 20


def test_prewarm_cache_spectral():
    from manylatents.experiment import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(10, 2).astype(np.float32)
    A = rng.randn(10, 10)
    A = A @ A.T

    class FakeModule:
        def affinity_matrix(self, use_symmetric=False):
            return A

    class FakeDataset:
        data = rng.randn(10, 5).astype(np.float32)

    cfgs = _make_metric_cfgs({
        "module.spectral_gap_ratio": {
            "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
        },
    })
    cache = prewarm_cache(cfgs, emb, FakeDataset(), module=FakeModule())
    assert "eigenvalues" in cache


def test_prewarm_cache_no_data_attribute():
    """prewarm_cache should not crash if dataset lacks .data."""
    from manylatents.experiment import prewarm_cache
    rng = np.random.RandomState(42)
    emb = rng.randn(20, 2).astype(np.float32)

    class NoDataDataset:
        pass

    cfgs = _make_metric_cfgs({
        "embedding.knn_preservation": {
            "_target_": "manylatents.metrics.knn_preservation.KNNPreservation",
            "n_neighbors": 5,
        },
    })
    cache = prewarm_cache(cfgs, emb, NoDataDataset())
    assert _content_key(emb) in cache  # embedding kNN still warmed
