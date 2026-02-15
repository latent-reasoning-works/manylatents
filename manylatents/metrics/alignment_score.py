"""Composite alignment scores and stratification utilities.

Combines multiple alignment metrics into per-variant scores for
stratification into aligned/divergent groups.
"""

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.metrics.cross_modal_jaccard import CrossModalJaccard
from manylatents.metrics.rank_agreement import RankAgreement


@dataclass
class StratificationResult:
    """Result of variant stratification by alignment score."""

    scores: np.ndarray  # Per-variant alignment scores
    strata: np.ndarray  # Stratum assignments: 0=divergent, 1=middle, 2=aligned
    thresholds: Tuple[float, float]  # (low, high) thresholds
    counts: Dict[str, int]  # Counts per stratum

    @property
    def aligned_mask(self) -> np.ndarray:
        """Boolean mask for aligned variants (top quartile)."""
        return self.strata == 2

    @property
    def divergent_mask(self) -> np.ndarray:
        """Boolean mask for divergent variants (bottom quartile)."""
        return self.strata == 0

    @property
    def middle_mask(self) -> np.ndarray:
        """Boolean mask for middle variants."""
        return self.strata == 1


def stratify_by_percentile(
    scores: np.ndarray,
    low_percentile: float = 25,
    high_percentile: float = 75,
) -> StratificationResult:
    """Stratify variants into aligned/middle/divergent by percentile thresholds.

    Args:
        scores: Per-variant alignment scores (higher = more aligned).
        low_percentile: Percentile threshold for divergent (default 25).
        high_percentile: Percentile threshold for aligned (default 75).

    Returns:
        StratificationResult with scores, strata assignments, and counts.
    """
    low_thresh = np.percentile(scores, low_percentile)
    high_thresh = np.percentile(scores, high_percentile)

    strata = np.ones(len(scores), dtype=int)  # Default to middle (1)
    strata[scores <= low_thresh] = 0  # Divergent
    strata[scores >= high_thresh] = 2  # Aligned

    counts = {
        "divergent": int((strata == 0).sum()),
        "middle": int((strata == 1).sum()),
        "aligned": int((strata == 2).sum()),
    }

    return StratificationResult(
        scores=scores,
        strata=strata,
        thresholds=(low_thresh, high_thresh),
        counts=counts,
    )


@register_metric(
    aliases=["alignment", "modal_alignment"],
    default_params={"method": "jaccard", "k": 20},
    description="Composite per-sample alignment score across modalities",
)
def AlignmentScore(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    k: int = 20,
    method: str = "jaccard",
    stratify: bool = False,
    low_percentile: float = 25,
    high_percentile: float = 75,
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray, StratificationResult]:
    """Compute composite per-variant alignment score.

    Combines pairwise alignment metrics into a single per-variant score.
    Higher scores indicate variants where modalities agree on local structure.

    Args:
        embeddings: Dict mapping modality names to embedding arrays.
        dataset: Optional dataset object (for protocol).
        module: Optional module object (for protocol).
        k: Number of neighbors for alignment computation.
        method: Alignment method ("jaccard" or "rank_lid").
        stratify: If True, return StratificationResult instead of raw scores.
        low_percentile: Percentile for divergent threshold (if stratify=True).
        high_percentile: Percentile for aligned threshold (if stratify=True).

    Returns:
        If stratify=False: Array of shape (N,) with per-variant alignment scores.
        If stratify=True: StratificationResult with scores and strata.
    """
    if method == "jaccard":
        scores = CrossModalJaccard(
            embeddings,
            dataset=dataset,
            module=module,
            k=k,
            return_per_sample=True,
        )
    elif method == "rank_lid":
        scores = RankAgreement(
            embeddings,
            dataset=dataset,
            module=module,
            k=k,
            metric_fn="lid",
            return_per_sample=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jaccard' or 'rank_lid'.")

    # Handle single array case
    if isinstance(scores, (int, float)):
        scores = np.array([scores])

    if stratify:
        return stratify_by_percentile(
            scores,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

    return scores
