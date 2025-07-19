import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule
from src.utils.metrics import (
    compute_average_smoothness,
    compute_geodesic_distances,
    compute_knn_laplacian,
    haversine_vectorized,
)

logger = logging.getLogger(__name__)

##############################################################################
# 1) Core metric: Spearman correlation of distances
##############################################################################

def preservation_metric(gt_dists, ac_dists, num_dists=50000, only_far=False):
    """
    Spearman correlation between two distance arrays (flattened).
    Optionally sample for performance, optionally only consider 
    "far" pairs (above 10th percentile).
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
# 2) Geographic Metric
##############################################################################

def compute_geographic_metric(
    ancestry_coords,
    latitude,
    longitude,
    use_medians=False,
    only_far=False,
    subset_to_test_on=None
):
    """
    Compare ground-truth haversine distances (lat/lon) vs. embedding distances (ancestry_coords).
    Return Spearman correlation.
    If use_medians=True, group by lat/lon and take median embedding coords.
    If only_far=True, ignore pairs under the 10th percentile of geo distance.
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        latitude = latitude[subset_to_test_on]
        longitude = longitude[subset_to_test_on]

    # Combine lat/lon in a DataFrame
    ground_truth_coords = pd.concat([latitude, longitude], axis=1)
    ground_truth_coords_rad = np.radians(ground_truth_coords)

    combined = pd.concat([
        ground_truth_coords_rad,
        pd.DataFrame(ancestry_coords, index=ground_truth_coords.index)
    ], axis=1)

    if use_medians:
        # group by lat/lon, median the embedding columns
        combined = combined.groupby(["latitude", "longitude"]).median().reset_index()

    # Extract arrays
    ground_truth_arr = combined[["latitude", "longitude"]].values
    ancestry_coords_arr = combined[[0, 1]].values  # two columns for embedding

    # Haversine distances => ground truth
    gt_dists_square = haversine_vectorized(ground_truth_arr)
    gt_dists = squareform(gt_dists_square)

    # Embedding (ancestry) distances
    ac_dists = pdist(ancestry_coords_arr)

    return preservation_metric(gt_dists, ac_dists, only_far=only_far)

## logging version for debugging
def _compute_geographic_metric(
    ancestry_coords,
    latitude,
    longitude,
    use_medians=False,
    only_far=False,
    subset_to_test_on=None,
):
    """
    Compare ground-truth haversine distances (lat/lon) vs. embedding distances (ancestry_coords).
    Return Spearman correlation.
    If use_medians=True, group by lat/lon and take median embedding coords.
    If only_far=True, ignore pairs under the 10th percentile of geo distance.
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        latitude = latitude[subset_to_test_on]
        longitude = longitude[subset_to_test_on]

    # Combine lat/lon in a DataFrame.
    ground_truth_coords = pd.concat([latitude, longitude], axis=1)
    logger.info(f"Combined ground_truth_coords shape before dropna: {ground_truth_coords.shape}")
    
    # Instead of dropping na and losing alignment, create a boolean mask.
    mask = ground_truth_coords.notna().all(axis=1)
    ground_truth_coords = ground_truth_coords[mask]
    logger.info(f"Combined ground_truth_coords shape after dropna: {ground_truth_coords.shape}")
    logger.info(f"Combined ground_truth_coords head (after dropna):\n{ground_truth_coords.head()}")

    # Filter the ancestry_coords with the same mask.
    # Convert the mask to a boolean numpy array.
    ancestry_coords = ancestry_coords[mask.values]

    # Now convert the cleaned ground truth coordinates to radians.
    ground_truth_coords_rad = np.radians(ground_truth_coords)

    # Combine the radian-converted ground truth with the corresponding ancestry coordinates.
    combined = pd.concat([
        ground_truth_coords_rad,
        pd.DataFrame(ancestry_coords, index=ground_truth_coords.index)
    ], axis=1)
    logger.info(f"Combined array shape: {combined.shape}")

    if use_medians:
        # Group by lat/lon and take median of embedding columns.
        combined = combined.groupby(["latitude", "longitude"]).median().reset_index()
        logger.info("After grouping by medians:")
        logger.info(f"Combined shape: {combined.shape}")

    # Extract arrays for distance computation.
    ground_truth_arr = combined[["latitude", "longitude"]].values
    ancestry_coords_arr = combined[[0, 1]].values  # two columns for embedding

    # Compute haversine distances (ground truth).
    gt_dists_square = haversine_vectorized(ground_truth_arr)
    logger.info(f"Initial gt_dists_square shape: {gt_dists_square.shape}")

    # Check symmetry before enforcing it.
    symmetry_diff = np.abs(gt_dists_square - gt_dists_square.T)
    logger.info(f"Max symmetry difference before forcing: {np.nanmax(symmetry_diff)}")

    # Enforce symmetry by averaging with its transpose.
    gt_dists_square = (gt_dists_square + gt_dists_square.T) / 2
    symmetry_diff = np.abs(gt_dists_square - gt_dists_square.T)
    logger.info(f"Max symmetry difference after forcing: {np.nanmax(symmetry_diff)}")

    # Verify the matrix is square.
    assert gt_dists_square.shape[0] == gt_dists_square.shape[1], "Distance matrix is not square!"

    # Convert the symmetric distance matrix to condensed form.
    gt_dists = squareform(gt_dists_square)
    # Compute the distances in the embedding space.
    ac_dists = pdist(ancestry_coords_arr)

    return preservation_metric(gt_dists, ac_dists, only_far=only_far)

##############################################################################
# 3) Admixture-based Metrics
##############################################################################

def compute_continental_admixture_metric_dists(
    ancestry_coords,
    admixture_ratios,
    population_label,
    use_medians=False,
    only_far=False,
    subset_to_test_on=None
):
    """
    Build geodesic distance graph for admixture_ratios, compare vs. ancestry_coords distances.
    If disconnected at k=5, increase k in steps of 5 up to 100. Return Spearman correlation.
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        admixture_ratios = admixture_ratios[subset_to_test_on]
        population_label = population_label[subset_to_test_on]

    df1 = pd.DataFrame(ancestry_coords, 
                       index=population_label.index)
    df1 = df1.rename(columns={i: f'emb{i}' for i in range(ancestry_coords.shape[1])})

    df2 = admixture_ratios.set_index(0).dropna()
    df2 = df2.rename(columns={i: f'ar{i}' for i in range(admixture_ratios.shape[1])}) # drop NA rows

    #df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    #df = pd.merge(df, population_label, left_index=True, right_index=True, how='inner')
    
    df = pd.concat([df1, df2.reset_index()], axis=1).drop(columns=0)
    df = pd.concat([df, population_label], axis=1)

    if use_medians:
        df = df.groupby(population_label.name).median().reset_index()

    ancestry_coords2 = df[df.columns[df.columns.map(str).str.startswith('emb')]].values
    admixture_ratios2 = df[df.columns[df.columns.map(str).str.startswith('ar')]].values

    ancestry_dists = pdist(ancestry_coords2)

    k = 5
    while k <= 100:
        admixture_dists = compute_geodesic_distances(admixture_ratios2, k=k, metric='euclidean')
        if admixture_dists is None:
            k += 5
        else:
            return preservation_metric(admixture_dists, ancestry_dists, only_far=only_far)

    return None  # if never got a connected graph


def compute_continental_admixture_metric_laplacian(
    ancestry_coords,
    admixture_ratios,
    subset_to_test_on=None
):
    """
    Evaluate smoothness x^T L x over adjacency built from ancestry_coords. 
    Return average across admixture components.
    """
    if subset_to_test_on is not None:
        ancestry_coords = ancestry_coords[subset_to_test_on]
        admixture_ratios = admixture_ratios[subset_to_test_on]

    laplacian = compute_knn_laplacian(ancestry_coords, k=5, normalized=True)
    return compute_average_smoothness(laplacian, admixture_ratios)

##############################################################################
# 4) Ground Truth based metrics
##############################################################################

def compute_ground_truth_preservation(ancestry_coords,
                                      gt_dists,
                                      **kwargs):
    
    gt_dists = gt_dists[np.triu_indices(gt_dists.shape[0], k=1)]
    ac_dists = pdist(ancestry_coords)
    return preservation_metric(gt_dists, 
                               ac_dists, 
                               **kwargs)


##############################################################################
# 5) Normalization helper
##############################################################################
def scale_embs(embs):
    # Make sure each component is at the same level
    return embs.values / np.abs(embs.values).max(axis=0)

##############################################################################
# 6) Example Aggregators
##############################################################################

def compute_k_admixture_metric_dists(
    ancestry_coords,
    admixtures_k,
    admixture_ratios_list,
    population_label,
    subset_to_test_on=None
):
    """
    For multiple 'k' admixture ratio sets, run compute_continental_admixture_metric_dists
    and return a dict of results keyed by k.
    """
    results = {}
    for k_val, admixture_ratios_k in zip(admixtures_k, admixture_ratios_list):
        key = f"admixture_preservation_k={k_val}"

        results[key] = compute_continental_admixture_metric_dists(
            scale_embs(ancestry_coords),
            admixture_ratios_k,
            population_label,
            use_medians=False,
            only_far=False,
            subset_to_test_on=subset_to_test_on
        )
    return results


def compute_quality_metrics(
    ancestry_coords,
    latitude,
    longitude,
    admixtures_k,
    admixture_ratios_list,
    population_label,
    subset_to_test_on=None
):
    """
    High-level aggregator for multiple metrics, returns a dictionary.
    """
    metrics_dict = {
        "geographic_preservation": compute_geographic_metric(
            scale_embs(ancestry_coords), latitude, longitude, 
            use_medians=False, only_far=False,
            subset_to_test_on=subset_to_test_on
        ),
        "geographic_preservation_medians": compute_geographic_metric(
            scale_embs(ancestry_coords), latitude, longitude,
            use_medians=True, only_far=False,
            subset_to_test_on=subset_to_test_on
        ),
        "geographic_preservation_far": compute_geographic_metric(
            scale_embs(ancestry_coords), latitude, longitude,
            use_medians=False, only_far=True,
            subset_to_test_on=subset_to_test_on
        ),
    }

    # Admixture aggregator
    admixture_dict = compute_k_admixture_metric_dists(
        scale_embs(ancestry_coords),
        admixtures_k,
        admixture_ratios_list,
        population_label,
        subset_to_test_on=subset_to_test_on
    )
    metrics_dict.update(admixture_dict)

    return metrics_dict


##############################################################################
# 7) Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def GeographicPreservation(embeddings: np.ndarray,
                           dataset, 
                           module: Optional[DimensionalityReductionModule] = None,
                           **kwargs) -> float:
    """
    Minimal wrapper that passes extra keyword arguments to compute_geographic_metric.
    """
    #module = kwargs.pop('module', None)  # Pop but ignore or handle separately

    return compute_geographic_metric(
        ancestry_coords=scale_embs(embeddings),
        latitude=dataset.latitude,
        longitude=dataset.longitude,
        subset_to_test_on=dataset.geographic_preservation_indices,
        **kwargs
    )

def AdmixturePreservation(embeddings: np.ndarray,
                          dataset, 
                          module: Optional[DimensionalityReductionModule] = None,
                          **kwargs) -> float:
    """
    Another single-value wrapper returning Spearman correlation.
    """
    return compute_continental_admixture_metric_dists(
        ancestry_coords=scale_embs(embeddings),
        admixture_ratios=dataset.admixture_ratios['5'],
        population_label=dataset.population_label,
        **kwargs
    )

def AdmixturePreservationK(embeddings: np.ndarray,
                           dataset, 
                           module: Optional[DimensionalityReductionModule] = None,
                           **kwargs) -> np.array:
    """
    A vector-value wrapper returning admixture preservation scores for all Ks.
    """
    
    return_vector = np.zeros(len(dataset.admixture_ratios))
    for i, key in enumerate(dataset.admixture_ratios.keys()):
        return_vector[i] = compute_continental_admixture_metric_dists(
            ancestry_coords=scale_embs(embeddings),
            admixture_ratios=dataset.admixture_ratios[key],
            population_label=dataset.population_label,
            **kwargs
        )
    return return_vector

def AdmixtureLaplacian(embeddings: np.ndarray,
                       dataset, 
                       module: Optional[DimensionalityReductionModule] = None) -> float:
    """
    Laplacian-based metric -> single float for callback usage.
    """
    return compute_continental_admixture_metric_laplacian(
        ancestry_coords=scale_embs(embeddings),
        admixture_ratios=dataset.admixture_ratios
    )

def GroundTruthPreservation(embeddings: np.ndarray, 
                            dataset, 
                            module: Optional[DimensionalityReductionModule] = None,
                            **kwargs) -> float:
    """
    Computes preservation of embedding distance versus ground truth distance (on synthetic data)
    Do not pass use_medians as a kwarg
    """

    assert hasattr(dataset, 'get_gt_dists')
    gt_dists = dataset.get_gt_dists()
    if "use_medians" in kwargs:
        raise ValueError("'use_medians' argument is not allowed.")

    return compute_ground_truth_preservation(scale_embs(embeddings),
                                             gt_dists,
                                             **kwargs)
