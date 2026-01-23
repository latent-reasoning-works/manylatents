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

__all__ = [
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
]
