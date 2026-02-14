"""Tests for merge_results enhancements."""
import numpy as np
import pandas as pd
import pytest


def test_generate_pivot_table():
    """generate_pivot_table produces expected format."""
    from manylatents.utils.merge_results import generate_pivot_table

    df = pd.DataFrame({
        "data": ["swissroll", "swissroll", "torus", "torus"],
        "algorithm": ["umap", "pca", "umap", "pca"],
        "trustworthiness": [0.95, 0.80, 0.90, 0.75],
        "seed": [42, 42, 42, 42],
    })
    pivot = generate_pivot_table(df, metric_cols=["trustworthiness"])
    assert "trustworthiness" in pivot.columns
    assert len(pivot) == 4  # 2 datasets x 2 algorithms


def test_parameter_sensitivity_summary():
    """parameter_sensitivity_summary computes mean/std across seeds."""
    from manylatents.utils.merge_results import parameter_sensitivity_summary

    df = pd.DataFrame({
        "n_neighbors": [5, 5, 10, 10],
        "min_dist": [0.1, 0.1, 0.1, 0.1],
        "trustworthiness": [0.90, 0.92, 0.88, 0.86],
        "seed": [42, 43, 42, 43],
    })
    summary = parameter_sensitivity_summary(
        df,
        param_cols=["n_neighbors", "min_dist"],
        metric_cols=["trustworthiness"],
    )
    assert "trustworthiness_mean" in summary.columns
    assert "trustworthiness_std" in summary.columns
    assert len(summary) == 2  # 2 unique (n_neighbors, min_dist) combos
