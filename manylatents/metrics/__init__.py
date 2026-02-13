"""Metrics for evaluating embeddings.

This module provides metrics for assessing dimensionality reduction quality.
Metrics are auto-registered via decorators and can be accessed by name or alias.

Usage:
    from manylatents.metrics import compute_metric, list_metrics

    # Compute a metric by alias
    beta_0 = compute_metric("beta_0", embeddings)
    pr = compute_metric("participation_ratio", embeddings)

    # List all available metrics
    print(list_metrics())

kNN Cache:
    Many metrics compute k-nearest neighbors. When running multiple metrics,
    use the shared kNN cache to avoid redundant computation:

    from manylatents.metrics import KNNCache
    # Cache is (distances, indices) tuple, shape (n_samples, max_k+1)
    # Metrics slice to their k: distances[:, 1:k+1]
"""

# Import registry first (no dependencies on other metrics)
from manylatents.metrics.registry import (
    compute_metric,
    get_metric,
    get_metric_registry,
    list_metrics,
    register_metric,
    resolve_metric,
    MetricSpec,
)

# Import metric protocol and types
from manylatents.metrics.metric import Metric, KNNCache

# Import all metric modules to trigger registration
# Core G-vector metrics
from manylatents.metrics.persistent_homology import PersistentHomology
from manylatents.metrics.participation_ratio import ParticipationRatio
from manylatents.metrics.lid import LocalIntrinsicDimensionality
from manylatents.metrics.trustworthiness import Trustworthiness

# Additional metrics (not yet decorated, but available via class name)
from manylatents.metrics.continuity import Continuity
from manylatents.metrics.anisotropy import Anisotropy
from manylatents.metrics.fractal_dimension import FractalDimension
from manylatents.metrics.correlation import PearsonCorrelation
from manylatents.metrics.knn_preservation import KNNPreservation
from manylatents.metrics.tangent_space import TangentSpaceApproximation
from manylatents.metrics.auc import AUC
from manylatents.metrics.outlier_score import OutlierScore

# Cross-modal alignment metrics
from manylatents.metrics.cka import CKA, cka_pairwise
from manylatents.metrics.cross_modal_jaccard import CrossModalJaccard, cross_modal_jaccard_pairwise
from manylatents.metrics.rank_agreement import RankAgreement
from manylatents.metrics.alignment_score import AlignmentScore, StratificationResult, stratify_by_percentile

# Spectral metrics
from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
from manylatents.metrics.spectral_decay_rate import SpectralDecayRate

# Embedding quality metrics
from manylatents.metrics.silhouette import SilhouetteScore

# Dataset metrics
from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

__all__ = [
    # Types
    "Metric",
    "KNNCache",
    # Registry functions
    "compute_metric",
    "get_metric",
    "get_metric_registry",
    "list_metrics",
    "register_metric",
    "resolve_metric",
    "MetricSpec",
    # Core metrics
    "PersistentHomology",
    "ParticipationRatio",
    "LocalIntrinsicDimensionality",
    "Trustworthiness",
    # Additional metrics
    "Continuity",
    "Anisotropy",
    "FractalDimension",
    "PearsonCorrelation",
    "KNNPreservation",
    "TangentSpaceApproximation",
    "AUC",
    "OutlierScore",
    # Cross-modal alignment metrics
    "CKA",
    "cka_pairwise",
    "CrossModalJaccard",
    "cross_modal_jaccard_pairwise",
    "RankAgreement",
    "AlignmentScore",
    "StratificationResult",
    "stratify_by_percentile",
    # Spectral metrics
    "SpectralGapRatio",
    "SpectralDecayRate",
    # Embedding quality metrics
    "SilhouetteScore",
    # Dataset metrics
    "GeodesicDistanceCorrelation",
]
