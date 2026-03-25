import numpy as np
import pytest
from sklearn.datasets import make_blobs


def test_loglog_consistency_returns_expected_keys():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    X, _ = make_blobs(n_samples=100, n_features=5, random_state=42)
    result = LogLogConsistency(X.astype(np.float32), k=50, k_min=5, k_steps=10)
    expected_keys = {
        "mean_r_squared", "std_r_squared", "frac_reliable", "mean_slope",
        "slope", "r_squared", "k_values", "mean_log_T",
    }
    assert set(result.keys()) == expected_keys


def test_loglog_consistency_shapes():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    n = 80
    X, _ = make_blobs(n_samples=n, n_features=3, random_state=42)
    result = LogLogConsistency(X.astype(np.float32), k=50, k_min=5, k_steps=10)
    assert result["slope"].shape == (n,)
    assert result["r_squared"].shape == (n,)
    assert len(result["k_values"]) == len(result["mean_log_T"])
    assert len(result["k_values"]) <= 10


def test_loglog_consistency_r_squared_bounds():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    X, _ = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)
    result = LogLogConsistency(X.astype(np.float32), k=50, k_min=5, k_steps=10)
    assert np.all(result["r_squared"] >= 0.0)
    assert np.all(result["r_squared"] <= 1.0)
    assert 0.0 <= result["mean_r_squared"] <= 1.0
    assert 0.0 <= result["frac_reliable"] <= 1.0


def test_loglog_consistency_positive_slope_on_blob():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    X, _ = make_blobs(n_samples=200, n_features=5, centers=1, random_state=42)
    result = LogLogConsistency(X.astype(np.float32), k=100, k_min=5, k_steps=15)
    assert result["mean_slope"] > 0


def test_loglog_consistency_duplicate_points():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    X = np.zeros((50, 3), dtype=np.float32)
    result = LogLogConsistency(X, k=30, k_min=5, k_steps=10)
    assert not np.any(np.isnan(result["r_squared"]))
    assert not np.any(np.isnan(result["slope"]))


def test_loglog_consistency_uses_cache():
    from manylatents.metrics.loglog_consistency import LogLogConsistency
    X, _ = make_blobs(n_samples=50, n_features=3, random_state=42)
    cache = {}
    LogLogConsistency(X.astype(np.float32), k=30, k_min=5, k_steps=10, cache=cache)
    assert len(cache) > 0


def test_loglog_consistency_registered():
    from manylatents.metrics import list_metrics
    names = list_metrics()
    assert "loglog_consistency" in names
    assert "lid_reliability" in names
