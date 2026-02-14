"""Tests for MetricAgreement post-hoc analysis."""
import numpy as np
import pandas as pd
import pytest


def test_metric_agreement_returns_correlation_matrix():
    """MetricAgreement returns pairwise Spearman correlation matrix."""
    from manylatents.metrics.metric_agreement import MetricAgreement

    df = pd.DataFrame({
        "trustworthiness": [0.95, 0.80, 0.70, 0.60],
        "continuity": [0.90, 0.75, 0.65, 0.55],
        "knn_preservation": [0.50, 0.85, 0.90, 0.30],
    })
    result = MetricAgreement(df, metric_cols=["trustworthiness", "continuity", "knn_preservation"])
    assert result.shape == (3, 3)
    # Diagonal should be 1.0
    np.testing.assert_allclose(np.diag(result.values), 1.0)


def test_metric_agreement_two_metrics():
    """MetricAgreement handles exactly 2 metrics."""
    from manylatents.metrics.metric_agreement import MetricAgreement

    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
    })
    result = MetricAgreement(df, metric_cols=["a", "b"])
    assert result.shape == (2, 2)
    np.testing.assert_allclose(np.diag(result.values), 1.0)
    # Perfect negative correlation
    assert result.loc["a", "b"] < 0
