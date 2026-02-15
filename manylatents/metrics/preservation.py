"""
Generic preservation metrics for dimensionality reduction.

This module provides core preservation metrics that work with any dataset
having ground truth distances. Domain-specific metrics (geography, admixture, etc.)
are in the manylatents-omics extension.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


##############################################################################
# Core metric: Spearman correlation of distances
##############################################################################

def preservation_metric(gt_dists, ac_dists, num_dists=50000, only_far=False):
    """
    Spearman correlation between two distance arrays (flattened).
    Optionally sample for performance, optionally only consider
    "far" pairs (above 10th percentile).

    Parameters
    ----------
    gt_dists : np.ndarray
        Ground truth distances (flattened)
    ac_dists : np.ndarray
        Actual/embedding distances (flattened)
    num_dists : int, optional
        Maximum number of distance pairs to sample (default: 50000)
    only_far : bool, optional
        If True, only consider pairs above 10th percentile (default: False)

    Returns
    -------
    float
        Spearman correlation coefficient
    """
    if only_far:
        cutoff = np.percentile(gt_dists, 10)
        mask = gt_dists >= cutoff
        gt_dists = gt_dists[mask]
        ac_dists = ac_dists[mask]

    # Subsample
    subset = np.random.choice(len(ac_dists), min(num_dists, len(ac_dists)), replace=False)
    corr, _ = spearmanr(gt_dists[subset], ac_dists[subset])
    return corr


##############################################################################
# Helper function for embedding scaling
##############################################################################

def _scale_embedding_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """
    Scale each embedding dimension to [0, 1] using min-max normalization.

    This ensures that no single dimension dominates distance calculations
    due to scale differences (e.g., UMAP dim 1: [-10, 10] vs dim 2: [-1, 1]).

    Parameters
    ----------
    embeddings : np.ndarray
        Input embeddings to rescale

    Returns
    -------
    np.ndarray
        Embeddings with all dimensions scaled to [0, 1]
    """
    embeddings = np.asarray(embeddings)

    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)
    emb_range = emb_max - emb_min

    # Avoid division by zero for constant dimensions
    emb_range = np.where(emb_range == 0, 1, emb_range)

    # Min-max normalization: (x - min) / (max - min)
    scaled_embeddings = (embeddings - emb_min) / emb_range

    return scaled_embeddings


##############################################################################
# Ground Truth based metrics
##############################################################################

def compute_ground_truth_preservation(ancestry_coords,
                                      gt_dists,
                                      **kwargs):
    """
    Compare embedding distances to ground truth distances.

    Parameters
    ----------
    ancestry_coords : np.ndarray
        Embedding coordinates (n_samples × n_components)
    gt_dists : np.ndarray
        Ground truth distance matrix (n_samples × n_samples)
    **kwargs
        Additional arguments passed to preservation_metric
        (e.g., num_dists, only_far)

    Returns
    -------
    float
        Spearman correlation between ground truth and embedding distances
    """
    gt_dists = gt_dists[np.triu_indices(gt_dists.shape[0], k=1)]
    ac_dists = pdist(ancestry_coords)
    return preservation_metric(gt_dists,
                               ac_dists,
                               **kwargs)


##############################################################################
# Single-Value Wrapper (conforms to Metric Protocol)
##############################################################################

@register_metric(
    aliases=["admixture_laplacian"],
    default_params={"scale_embeddings": True},
    description="Admixture Laplacian preservation score",
)
def GroundTruthPreservation(embeddings: np.ndarray,
                            dataset,
                            module: Optional[LatentModule] = None,
                            scale_embeddings: bool = True,
                            **kwargs) -> float:
    """
    Computes preservation of embedding distance versus ground truth distance.

    This metric is generic and works with any dataset that provides a
    `get_gt_dists()` method returning ground truth pairwise distances.

    Parameters
    ----------
    embeddings : np.ndarray
        Low-dimensional embeddings (n_samples × n_components)
    dataset : object
        Dataset object with `get_gt_dists()` method
    module : LatentModule, optional
        Algorithm module (unused, for protocol compatibility)
    scale_embeddings : bool, optional
        Whether to scale embedding dimensions to [0, 1] (default: True)
    **kwargs
        Additional arguments passed to preservation_metric
        (e.g., only_far, num_dists). Do NOT pass 'use_medians'.

    Returns
    -------
    float
        Spearman correlation between ground truth and embedding distances

    Raises
    ------
    ValueError
        If 'use_medians' is passed in kwargs (not supported for this metric)
    AssertionError
        If dataset does not have `get_gt_dists()` method
    """
    if scale_embeddings:
        embeddings = _scale_embedding_dimensions(embeddings)

    assert hasattr(dataset, 'get_gt_dists'), \
        "Dataset must have get_gt_dists() method for ground truth preservation"
    gt_dists = dataset.get_gt_dists()

    if "use_medians" in kwargs:
        raise ValueError("'use_medians' argument is not allowed.")

    return compute_ground_truth_preservation(embeddings,
                                             gt_dists,
                                             **kwargs)
