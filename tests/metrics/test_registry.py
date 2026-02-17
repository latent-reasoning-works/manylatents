# tests/metrics/test_registry.py
"""Tests for the metric registry."""
import importlib

import numpy as np
import pytest

_has_ripser = importlib.util.find_spec("ripser") is not None
needs_ripser = pytest.mark.skipif(not _has_ripser, reason="ripser not installed")


@needs_ripser
def test_registry_has_beta_aliases():
    """Registry contains beta_0 and beta_1 aliases."""
    from manylatents.metrics import get_metric_registry

    registry = get_metric_registry()

    assert "beta_0" in registry
    assert "beta_1" in registry
    assert registry["beta_0"].func.__name__ == "PersistentHomology"
    assert registry["beta_0"].params == {"homology_dim": 0}
    assert registry["beta_1"].params == {"homology_dim": 1}


def test_registry_has_participation_ratio():
    """Registry contains participation_ratio alias."""
    from manylatents.metrics import get_metric_registry

    registry = get_metric_registry()

    assert "participation_ratio" in registry
    assert "pr" in registry
    assert registry["participation_ratio"].func.__name__ == "ParticipationRatio"


def test_registry_has_lid():
    """Registry contains local intrinsic dimensionality aliases."""
    from manylatents.metrics import get_metric_registry

    registry = get_metric_registry()

    assert "local_intrinsic_dim" in registry
    assert "lid" in registry
    assert registry["lid"].func.__name__ == "LocalIntrinsicDimensionality"


@needs_ripser
def test_resolve_metric():
    """resolve_metric returns function and params."""
    from manylatents.metrics import resolve_metric

    fn, params = resolve_metric("beta_0")
    assert fn.__name__ == "PersistentHomology"
    assert params == {"homology_dim": 0}

    fn, params = resolve_metric("beta_1")
    assert params == {"homology_dim": 1}


@needs_ripser
def test_list_metrics():
    """list_metrics returns sorted list of all metric names."""
    from manylatents.metrics import list_metrics

    metrics = list_metrics()

    assert isinstance(metrics, list)
    assert "beta_0" in metrics
    assert "beta_1" in metrics
    assert "participation_ratio" in metrics
    assert "local_intrinsic_dim" in metrics
    assert metrics == sorted(metrics)  # Sorted


@needs_ripser
def test_get_metric():
    """get_metric returns callable MetricSpec."""
    from manylatents.metrics import get_metric

    spec = get_metric("beta_0")
    assert callable(spec)
    assert spec.params == {"homology_dim": 0}


def test_get_metric_raises_on_unknown():
    """get_metric raises KeyError for unknown metric."""
    from manylatents.metrics import get_metric

    with pytest.raises(KeyError, match="not found"):
        get_metric("nonexistent_metric")


@needs_ripser
def test_compute_metric_beta_0():
    """compute_metric works for beta_0."""
    from manylatents.metrics import compute_metric

    # Simple 2D point cloud - should have 1 connected component
    embeddings = np.random.randn(100, 2)

    result = compute_metric("beta_0", embeddings)

    assert isinstance(result, (int, float))
    assert result >= 1  # At least one connected component


def test_compute_metric_participation_ratio():
    """compute_metric works for participation_ratio."""
    from manylatents.metrics import compute_metric

    embeddings = np.random.randn(100, 10)

    result = compute_metric("participation_ratio", embeddings)

    assert isinstance(result, float)
    assert result > 0


def test_compute_metric_lid():
    """compute_metric works for local_intrinsic_dim."""
    from manylatents.metrics import compute_metric

    embeddings = np.random.randn(100, 10)

    result = compute_metric("local_intrinsic_dim", embeddings)

    assert isinstance(result, float)
    assert result > 0
