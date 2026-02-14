"""Metric Agreement analysis.

Computes pairwise Spearman rank correlation between metrics across runs.
This is a post-hoc analysis tool, not a standard embedding metric.
"""
import pandas as pd
from scipy.stats import spearmanr


def MetricAgreement(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between metrics across runs.

    Args:
        df: DataFrame with metric columns.
        metric_cols: Column names of metrics to compare.

    Returns:
        DataFrame: Symmetric correlation matrix.
    """
    subset = df[metric_cols].dropna()
    corr_matrix, _ = spearmanr(subset.values)
    if len(metric_cols) == 2:
        corr_matrix = [[1.0, corr_matrix], [corr_matrix, 1.0]]
    return pd.DataFrame(corr_matrix, index=metric_cols, columns=metric_cols)
