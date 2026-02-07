"""Rank-based comparison of per-sample metrics across modalities.

Uses percentile ranks to compare LID/PR values across modalities with
different ambient dimensions, sidestepping the dimension mismatch problem.
"""

from typing import Dict, Optional, Union
import numpy as np
from scipy.stats import rankdata, spearmanr

from manylatents.metrics.registry import register_metric
from manylatents.metrics.lid import LocalIntrinsicDimensionality
from manylatents.metrics.participation_ratio import ParticipationRatio


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D, squeezing if needed."""
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.squeeze(1)
    return arr


def compute_percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Convert values to percentile ranks (0-1 scale)."""
    ranks = rankdata(values, method="average")
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(ranks)


@register_metric(
    aliases=["lid_rank_agreement", "rank_correlation"],
    default_params={"metric_fn": "lid", "k": 20},
    description="Rank-based agreement of LID/PR across modalities",
)
def RankAgreement(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    k: int = 20,
    metric_fn: str = "lid",
    return_correlations: bool = False,
    return_per_sample: bool = False,
) -> Union[float, np.ndarray, Dict[str, float]]:
    """Compare per-sample metrics across modalities using percentile ranks.

    Converts metric values (LID or PR) to percentile ranks within each modality,
    then compares ranks. This sidesteps ambient dimension mismatch since we
    compare relative orderings, not absolute values.

    Args:
        embeddings: Either single array or dict mapping modality names to arrays.
        dataset: Optional dataset object (unused, for protocol).
        module: Optional module object (unused, for protocol).
        k: Number of neighbors for metric computation.
        metric_fn: Which metric to use ("lid" or "pr").
        return_correlations: If True, return pairwise Spearman correlations.
        return_per_sample: If True, return per-sample rank agreement scores.

    Returns:
        If return_correlations: Dict mapping pair names to Spearman rho.
        If return_per_sample: Array of shape (N,) with mean rank agreement.
        Otherwise: Scalar mean Spearman correlation across all pairs.
    """
    # Single array case
    if isinstance(embeddings, np.ndarray):
        n = _ensure_2d(embeddings).shape[0]
        if return_per_sample:
            return np.ones(n)
        if return_correlations:
            return {"self": 1.0}
        return 1.0

    # Multi-modal case
    modality_names = list(embeddings.keys())
    n_modalities = len(modality_names)

    if n_modalities < 2:
        raise ValueError("Need at least 2 modalities for rank comparison")

    # Compute metric values for each modality
    metric_values = {}
    for name, emb in embeddings.items():
        emb = _ensure_2d(emb)
        if metric_fn == "lid":
            vals = LocalIntrinsicDimensionality(
                emb, dataset=None, module=None, k=k, return_per_sample=True
            )
        elif metric_fn == "pr":
            vals = ParticipationRatio(
                emb, dataset=None, module=None, return_per_sample=True
            )
        else:
            raise ValueError(f"Unknown metric_fn: {metric_fn}. Use 'lid' or 'pr'.")
        metric_values[name] = vals

    # Convert to percentile ranks
    rank_values = {name: compute_percentile_ranks(vals) for name, vals in metric_values.items()}

    # Compute pairwise correlations and rank differences
    correlations = {}
    rank_diffs = {}

    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            name_a = modality_names[i]
            name_b = modality_names[j]
            pair_name = f"{name_a}_{name_b}"

            ranks_a = rank_values[name_a]
            ranks_b = rank_values[name_b]

            rho, _ = spearmanr(ranks_a, ranks_b)
            correlations[pair_name] = rho
            rank_diffs[pair_name] = 1.0 - np.abs(ranks_a - ranks_b)

    if return_correlations:
        return correlations

    if return_per_sample:
        all_diffs = np.stack(list(rank_diffs.values()), axis=0)
        return all_diffs.mean(axis=0)

    return float(np.mean(list(correlations.values())))
