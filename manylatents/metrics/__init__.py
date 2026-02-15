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

Cache:
    Many metrics compute k-nearest neighbors or eigenvalues. Pass a shared
    cache dict to avoid redundant computation:

    cache = {}
    metric_fn(embeddings, dataset, module, cache=cache)
    # cache is populated by compute_knn() / compute_eigenvalues() internally.
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
from manylatents.metrics.metric import Metric

# Import all metric modules to trigger registration
# Core G-vector metrics
from manylatents.metrics.persistent_homology import PersistentHomology
from manylatents.metrics.participation_ratio import ParticipationRatio
from manylatents.metrics.lid import LocalIntrinsicDimensionality
from manylatents.metrics.trustworthiness import Trustworthiness

# Additional embedding metrics
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
from manylatents.metrics.affinity_spectrum import AffinitySpectrum

# Module metrics (kernel/graph-based)
from manylatents.metrics.connected_components import ConnectedComponents
from manylatents.metrics.diffusion_map_correlation import DiffusionMapCorrelation
from manylatents.metrics.kernel_matrix_sparsity import KernelMatrixSparsity, KernelMatrixDensity

# Diffusion-based metrics
from manylatents.metrics.diffusion_condensation import DiffusionCondensation
from manylatents.metrics.diffusion_curvature import DiffusionCurvature
from manylatents.metrics.diffusion_spectral_entropy import DiffusionSpectralEntropy

# Topological metrics (optional deps: gudhi, ripser)
try:
    from manylatents.metrics.reeb_graph import ReebGraphNodesEdges
except ImportError:
    pass

# Magnitude metrics (optional dep: magnipy)
try:
    from manylatents.metrics.magnitude_dimension import MagnitudeDimension
except ImportError:
    pass

# Embedding quality metrics
from manylatents.metrics.silhouette import SilhouetteScore

# Dataset metrics
from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation
from manylatents.metrics.dataset_topology_descriptor import DatasetTopologyDescriptor
from manylatents.metrics.preservation import GroundTruthPreservation
from manylatents.metrics.stratification import kmeans_stratification

# Post-hoc analysis
from manylatents.metrics.metric_agreement import MetricAgreement

__all__ = [
    # Types
    "Metric",
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
    "AffinitySpectrum",
    # Module metrics (kernel/graph-based)
    "ConnectedComponents",
    "DiffusionMapCorrelation",
    "KernelMatrixSparsity",
    "KernelMatrixDensity",
    # Diffusion-based metrics
    "DiffusionCondensation",
    "DiffusionCurvature",
    "DiffusionSpectralEntropy",
    # Topological metrics
    "ReebGraphNodesEdges",
    # Magnitude metrics
    "MagnitudeDimension",
    # Embedding quality metrics
    "SilhouetteScore",
    # Dataset metrics
    "GeodesicDistanceCorrelation",
    "DatasetTopologyDescriptor",
    "GroundTruthPreservation",
    "kmeans_stratification",
    # Post-hoc analysis
    "MetricAgreement",
]
