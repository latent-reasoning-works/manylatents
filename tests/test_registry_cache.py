"""Test that compute_metric passes cache through to metric functions."""
import numpy as np
from manylatents.metrics.registry import compute_metric, compute_metric_detailed


def test_compute_metric_passes_cache():
    """cache= kwarg reaches the underlying metric function."""
    received = {}

    def fake_metric(embeddings, dataset=None, module=None, cache=None, **kwargs):
        received["cache"] = cache
        return 1.0

    from manylatents.metrics.registry import _REGISTRY, MetricSpec
    _REGISTRY["_test_cache_metric"] = MetricSpec(func=fake_metric)
    try:
        cache = {"prewarmed": True}
        result = compute_metric("_test_cache_metric", np.zeros((5, 2)), cache=cache)
        assert result == 1.0
        assert received["cache"] is cache
    finally:
        del _REGISTRY["_test_cache_metric"]


def test_compute_metric_detailed_passes_cache():
    """cache= kwarg reaches the underlying metric function via detailed path."""
    received = {}

    def fake_metric(embeddings, dataset=None, module=None, cache=None, **kwargs):
        received["cache"] = cache
        return {"score": 0.5}

    from manylatents.metrics.registry import _REGISTRY, MetricSpec
    _REGISTRY["_test_cache_detailed"] = MetricSpec(func=fake_metric)
    try:
        cache = {"prewarmed": True}
        result = compute_metric_detailed("_test_cache_detailed", np.zeros((5, 2)), cache=cache)
        assert received["cache"] is cache
        assert result["raw"] == {"score": 0.5}
    finally:
        del _REGISTRY["_test_cache_detailed"]


def test_cache_none_by_default():
    """When no cache is passed, metric receives None."""
    received = {}

    def fake_metric(embeddings, dataset=None, module=None, cache=None, **kwargs):
        received["cache"] = cache
        return 1.0

    from manylatents.metrics.registry import _REGISTRY, MetricSpec
    _REGISTRY["_test_no_cache"] = MetricSpec(func=fake_metric)
    try:
        compute_metric("_test_no_cache", np.zeros((5, 2)))
        assert received["cache"] is None
    finally:
        del _REGISTRY["_test_no_cache"]
