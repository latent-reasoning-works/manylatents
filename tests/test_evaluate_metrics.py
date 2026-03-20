"""Test the Hydra-free evaluate_metrics() API."""
import numpy as np
import pytest


def test_evaluate_metrics_basic():
    """evaluate_metrics resolves metric names and returns flat dict."""
    from manylatents.evaluate import evaluate_metrics
    rng = np.random.RandomState(42)
    emb = rng.randn(50, 2).astype(np.float32)
    results = evaluate_metrics(emb, metrics=["FractalDimension"])
    assert isinstance(results, dict)
    assert len(results) > 0
    for v in results.values():
        assert isinstance(v, (int, float, np.integer, np.floating, np.ndarray))


def test_evaluate_metrics_shared_cache():
    """Cache is shared across metric calls -- kNN computed once."""
    from manylatents.evaluate import evaluate_metrics
    rng = np.random.RandomState(42)
    emb = rng.randn(50, 2).astype(np.float32)
    cache = {}
    evaluate_metrics(emb, metrics=["Anisotropy"], cache=cache)
    cache_size_after_first = len(cache)
    evaluate_metrics(emb, metrics=["Anisotropy"], cache=cache)
    assert len(cache) == cache_size_after_first


def test_evaluate_metrics_creates_cache_if_none():
    from manylatents.evaluate import evaluate_metrics
    rng = np.random.RandomState(42)
    emb = rng.randn(30, 2).astype(np.float32)
    results = evaluate_metrics(emb, metrics=["FractalDimension"])
    assert isinstance(results, dict)


def test_evaluate_metrics_unknown_metric():
    from manylatents.evaluate import evaluate_metrics
    with pytest.raises(KeyError):
        evaluate_metrics(np.zeros((10, 2)), metrics=["nonexistent_metric_xyz"])


def test_evaluate_metrics_dict_result_flattened():
    """Metrics returning dicts get flattened with dotted keys."""
    from manylatents.evaluate import evaluate_metrics
    from manylatents.metrics.registry import _REGISTRY, MetricSpec

    def fake_dict_metric(embeddings, dataset=None, module=None, cache=None, **kw):
        return {"sub_a": 1.0, "sub_b": 2.0}

    _REGISTRY["_test_dict_metric"] = MetricSpec(func=fake_dict_metric)
    try:
        results = evaluate_metrics(np.zeros((5, 2)), metrics=["_test_dict_metric"])
        assert "_test_dict_metric.sub_a" in results
        assert "_test_dict_metric.sub_b" in results
        assert results["_test_dict_metric.sub_a"] == 1.0
    finally:
        del _REGISTRY["_test_dict_metric"]


def test_evaluate_metrics_import_from_package():
    from manylatents import evaluate_metrics
    assert callable(evaluate_metrics)
