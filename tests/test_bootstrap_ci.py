"""Tests for manylatents.utils.stats.bootstrap_ci."""

import numpy as np
import pytest

from manylatents.utils.stats import bootstrap_ci


def test_basic_mean_ci():
    """CI around a known mean should contain the true mean."""
    rng = np.random.RandomState(0)
    data = rng.normal(loc=5.0, scale=1.0, size=500)
    lo, hi = bootstrap_ci(np.mean, data, n_bootstrap=500)
    assert lo < 5.0 < hi


def test_auroc_ci():
    """CI for a perfect classifier should be near 1.0."""
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.array([0.1] * 100 + [0.9] * 100)

    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(roc_auc_score, y_true, y_score, n_bootstrap=200)
    assert lo > 0.95
    assert hi <= 1.0


def test_correlation_ci():
    """CI for Spearman correlation on correlated data."""
    from scipy.stats import spearmanr
    rng = np.random.RandomState(42)
    x = rng.randn(200)
    y = x + rng.randn(200) * 0.3  # strong positive correlation

    def spearman_rho(a, b):
        return spearmanr(a, b).statistic

    lo, hi = bootstrap_ci(spearman_rho, x, y, n_bootstrap=300)
    assert lo > 0.5
    assert hi < 1.0


def test_multiple_arrays():
    """Works with 3+ parallel arrays."""
    a = np.arange(100, dtype=float)
    b = np.arange(100, dtype=float) * 2
    c = np.arange(100, dtype=float) * 3

    def sum_means(x, y, z):
        return np.mean(x) + np.mean(y) + np.mean(z)

    lo, hi = bootstrap_ci(sum_means, a, b, c)
    # True value: 49.5 + 99 + 148.5 = 297
    assert lo < 297 < hi


def test_seed_reproducibility():
    """Same seed → same CI."""
    data = np.random.RandomState(0).randn(100)
    ci1 = bootstrap_ci(np.mean, data, seed=123)
    ci2 = bootstrap_ci(np.mean, data, seed=123)
    assert ci1 == ci2


def test_different_seeds_differ():
    """Different seeds → different CIs."""
    data = np.random.RandomState(0).randn(100)
    ci1 = bootstrap_ci(np.mean, data, seed=1)
    ci2 = bootstrap_ci(np.mean, data, seed=2)
    assert ci1 != ci2


def test_ci_level():
    """Wider CI level → wider interval."""
    data = np.random.RandomState(0).randn(500)
    lo90, hi90 = bootstrap_ci(np.mean, data, ci=0.90, n_bootstrap=500)
    lo99, hi99 = bootstrap_ci(np.mean, data, ci=0.99, n_bootstrap=500)
    assert (hi99 - lo99) > (hi90 - lo90)


def test_empty_arrays_raises():
    bootstrap_ci_raises = pytest.raises(ValueError, match="At least one array")
    with bootstrap_ci_raises:
        bootstrap_ci(np.mean)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci(np.mean, np.arange(10), np.arange(5))


def test_failing_stat_fn_raises():
    """If stat_fn fails on most resamples, raise RuntimeError."""
    def always_fail(x):
        raise ValueError("nope")

    with pytest.raises(RuntimeError, match="bootstrap resamples"):
        bootstrap_ci(always_fail, np.arange(10, dtype=float), n_bootstrap=50)
