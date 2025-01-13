import os
import copy
import hashlib
import pickle
import pandas as pd
import seaborn as sns
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import phate
from sklearn.manifold import TSNE
import tqdm

import scprep
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import csr_matrix

# Cache directory for geodesic distances
CACHE_DIR = "./results/geodesic_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache directory for graph Laplacians
LAPLACIAN_CACHE_DIR = "./results/laplacian_cache"
os.makedirs(LAPLACIAN_CACHE_DIR, exist_ok=True)

# Compute quality metrics
def compute_quality_metrics(ancestry_coords, metadata, admixtures_k, admixture_ratios_list):
    to_keep = ~metadata['filter_pca_outlier'] & ~metadata['hard_filtered'] & ~metadata['filter_contaminated']
    ancestry_coords = ancestry_coords[to_keep]
    metadata = metadata[to_keep]
    admixture_ratios = [admixture_ratios_list_item[to_keep] for admixture_ratios_list_item in admixture_ratios_list]
    #admixture_ratios = admixture_ratios_list[3][to_keep]

    # geographic metrics
    metrics_dict = {
        "geographic_preservation": compute_geographic_metric(ancestry_coords, 
                                                                     metadata, 
                                                                     use_medians=False),
        "geographic_preservation_medians": compute_geographic_metric(ancestry_coords, 
                                                                             metadata, 
                                                                             use_medians=True),
        "geographic_preservation_far": compute_geographic_metric(ancestry_coords, 
                                                                         metadata, 
                                                                         use_medians=False, 
                                                                         only_far=True)
    }
    
    # admixture metrics
    for k, admixture_ratios_item in zip(admixtures_k, admixture_ratios):
        metrics_dict.update({
            "admixture_preservation_k={}".format(k): compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                        admixture_ratios_item, 
                                                                                                        metadata, 
                                                                                                        use_medians=False),
            "admixture_preservation_medians_k={}".format(k): compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                                admixture_ratios_item, 
                                                                                                                metadata, 
                                                                                                                use_medians=True),
            "admixture_preservation_far_k={}".format(k): compute_continental_admixture_metric_dists(ancestry_coords, 
                                                                                                            admixture_ratios_item, 
                                                                                                            metadata, 
                                                                                                            use_medians=False, 
                                                                                                            only_far=True),
            "admixture_preservation_laplacian_k={}".format(k): compute_continental_admixture_metric_laplacian(ancestry_coords, 
                                                                                                                      admixture_ratios_item),
        })

    return metrics_dict

def compute_pca_metrics(pca_input, emb, metadata):
    to_keep = ~metadata['filter_pca_outlier'] & ~metadata['hard_filtered'] & ~metadata['filter_contaminated']
    metrics_dict = {'pca_correlation': compute_pca_similarity(pca_input[to_keep], emb[to_keep])}

    return metrics_dict

def compute_topological_metrics(emb, metadata, phate_operator):
    # Adjacency matrix (iffusion operator, minus diagonal)
    A = phate_operator.diff_op - np.diag(phate_operator.diff_op)*np.eye(len(phate_operator.diff_op))
    graph = nx.from_numpy_array(A) # put into networkx
    component_Sizes = np.sort(np.array([len(k) for k in nx.connected_components(graph)]))[::-1]
    
    metrics_dict = {'connected_components': len(component_Sizes),
                    'component_sizes': component_Sizes}
    
    return metrics_dict

# Helper to compute and append metrics
def compute_and_append_metrics(method_name, emb, pca_input, metadata, admixtures_k, admixture_ratios_list, hyperparam_dict, operator, results):
    # Compute metrics
    metrics_dict = compute_quality_metrics(emb, metadata, admixtures_k, admixture_ratios_list)
    
    # Add empty topological metrics if not computed
    if method_name in ["pca (2D)", "pca (50D)", "t-SNE"]:
        topological_dict = {'connected_components': None, 
                            'component_sizes': None}
    else:
        topological_dict = compute_topological_metrics(emb, metadata, operator)

    pca_metric_dict = compute_pca_metrics(pca_input, emb, metadata)

    metrics_dict.update(pca_metric_dict)
    metrics_dict.update(topological_dict)
    metrics_dict.update(hyperparam_dict)
    metrics_dict.update({'method': method_name})


    
    results.append(metrics_dict)

def compute_geodesic_distances(embedding, k=10, metric='euclidean', verbose=0):
    """
    Compute geodesic distances on a KNN graph built from the embedding, with caching.

    Parameters:
    embedding (numpy.ndarray or pandas.DataFrame): An (N, D) array where N is the number of samples, D is the dimensionality.
    k (int): Number of nearest neighbors to use for the KNN graph.
    metric (str): Distance metric to use for the KNN graph (e.g., 'euclidean', 'manhattan').

    Returns:
    geodesic_distances (numpy.ndarray): An (N, N) geodesic distance matrix.
    """
    
    # Convert embedding to numpy array if it's a DataFrame
    if isinstance(embedding, pd.DataFrame):
        embedding = embedding.values
    
    # Create a unique hash for the input arguments
    embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"geodesic_{embedding_hash}_k={k}_metric={metric}.pkl")

    # Check if cached result exists
    if os.path.exists(cache_file):
        if verbose >= 1:
            print(f"Loading cached geodesic distances from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Compute geodesic distances
    if verbose >= 1:
        print(f"Computing geodesic distances for k={k}, metric={metric}")
    knn_graph = kneighbors_graph(embedding, n_neighbors=k, mode='distance', metric=metric, include_self=False)
    
    n_components, labels = connected_components(knn_graph, directed=False)
    if n_components > 1:
        if verbose >= 2:
            print(f"{n_components} components. The KNN graph is disconnected. Geodesic distances will contain infinities.")
        return None

    # Compute geodesic distances (shortest path on the graph)
    geodesic_distances = shortest_path(knn_graph, directed=False, method='D')  # 'D' is Dijkstra's algorithm

    # Extract the upper triangular part of the distance matrix as a flattened array
    flattened_dists = geodesic_distances[np.triu_indices_from(geodesic_distances, k=1)]

    # Save the result to the cache
    with open(cache_file, 'wb') as f:
        pickle.dump(flattened_dists, f)
    
    return flattened_dists


def haversine_vectorized(coords):
    """
    Compute pairwise haversine distances in a vectorized manner.
    coords: (n_samples, 2) array of [latitude, longitude] in radians.
    Returns:
        Square pairwise distance matrix.
    """
    lat = coords[:, 0][:, np.newaxis]  # Reshape for broadcasting
    lon = coords[:, 1][:, np.newaxis]

    # Compute differences
    dlat = lat - lat.T
    dlon = lon - lon.T

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius (6371 km)
    distances = 6371 * c

    return distances

def preservation_metric(gt_dists, ac_dists, num_dists=50000, only_far=False):
    """
    Compute the Spearman correlation between two pairwise distance matrices.
    gt_dists: Ground truth condensed distance matrix.
    ac_dists: Ancestry condensed distance matrix.
    num_samples: Number of distances to sample for Spearman correlation.
    """

    if only_far:
        cutoff = np.percentile(gt_dists, 10) # looking at points ~ not in the same cluster
        index = gt_dists >= cutoff
        gt_dists = gt_dists[index]
        ac_dists = ac_dists[index]

    # Take a random subset of distances for Spearman correlation
    subset = np.random.choice(len(ac_dists), min(num_dists, len(ac_dists)), replace=False)

    # Compute Spearman correlation
    corr, _ = spearmanr(gt_dists[subset], ac_dists[subset], axis=0)
    return corr

def compute_pca_similarity(pca_input, ancestry_coords):

    # compute distances for PCA and ancestry coords
    pca_dists = pdist(pca_input)
    ac_dists = pdist(ancestry_coords)

    return preservation_metric(pca_dists, ac_dists)

def compute_geographic_metric(ancestry_coords, merged_metadata, use_medians=False, only_far=False):
    # geography doesn't mean anything for these populations
    include_index = ~((merged_metadata['Genetic_region_merged'] == 'America') | merged_metadata['Population'].isin(['ACB', 'ASW', 'CEU']))
    ground_truth_coords = merged_metadata[['latitude', 'longitude']]
    ground_truth_coords_rad = np.radians(ground_truth_coords)

    summarized_df = pd.concat([ground_truth_coords_rad, 
                               pd.DataFrame(ancestry_coords, index=ground_truth_coords.index)],
                              axis=1)[include_index]

    if use_medians:
        summarized_df = summarized_df.groupby(['latitude', 'longitude']).median().reset_index()
    ground_truth_coords_input = summarized_df[['latitude', 'longitude']].values
    ancestry_coords_input = summarized_df[[0,1]]

    # compute distances and final metric
    ground_truth_dists_square = haversine_vectorized(ground_truth_coords_input)
    ground_truth_dists = squareform(ground_truth_dists_square)
    ancestry_dists = pdist(ancestry_coords_input)

    geographic_preservation = preservation_metric(ground_truth_dists, 
                                                  ancestry_dists,
                                                  only_far=only_far)
    return geographic_preservation


def compute_knn_laplacian(ancestry_coords, k=5, normalized=True, verbose=0):
    """
    Compute the graph Laplacian for a KNN graph based on ancestry coordinates, with caching.

    Parameters:
    ancestry_coords (numpy.ndarray): An (N, D) array of ancestry coordinates (N samples, D dimensions).
    k (int): Number of nearest neighbors for the KNN graph.
    normalized (bool): Whether to compute the normalized Laplacian. If False, returns the unnormalized Laplacian.
    verbose (int): Verbosity level for logging (0: silent, 1: log caching info, 2: detailed info).

    Returns:
    laplacian (numpy.ndarray): The graph Laplacian matrix (N, N).
    """
    # Convert ancestry_coords to numpy array if needed
    if isinstance(ancestry_coords, pd.DataFrame):
        ancestry_coords = ancestry_coords.values

    # Create a unique hash for the input arguments
    coords_hash = hashlib.sha256(ancestry_coords.tobytes()).hexdigest()
    cache_file = os.path.join(
        LAPLACIAN_CACHE_DIR, f"laplacian_{coords_hash}_k={k}_normalized={normalized}.pkl"
    )

    # Check if cached result exists
    if os.path.exists(cache_file):
        if verbose >= 1:
            print(f"Loading cached Laplacian from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Compute adjacency matrix for KNN graph
    if verbose >= 1:
        print(f"Computing KNN graph Laplacian for k={k}, normalized={normalized}")
    adjacency = kneighbors_graph(
        ancestry_coords, n_neighbors=k, mode='connectivity', include_self=False
    )
    adjacency = adjacency.toarray()

    # Degree matrix
    degree_matrix = np.diag(adjacency.sum(axis=1))

    # Compute Laplacian
    if normalized:
        # Normalized Laplacian
        with np.errstate(divide='ignore'):  # Handle divisions by zero for isolated nodes
            degree_inv_sqrt = np.linalg.pinv(np.sqrt(degree_matrix))
        laplacian = np.eye(adjacency.shape[0]) - degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    else:
        # Unnormalized Laplacian
        laplacian = degree_matrix - adjacency

    # Save the Laplacian to the cache
    with open(cache_file, 'wb') as f:
        pickle.dump(laplacian, f)

    if verbose >= 1:
        print(f"Laplacian cached at {cache_file}")

    return laplacian

def compute_average_smoothness(laplacian, admixture_ratios):
    """
    Compute the average smoothness of admixture ratios over a graph.

    Parameters:
    laplacian (numpy.ndarray): The graph Laplacian matrix (N, N).
    admixture_ratios (numpy.ndarray): An (N, C) array of admixture ratios (N samples, C components).

    Returns:
    average_smoothness (float): The average smoothness over all components.
    """
    smoothness_values = []
    for i in range(admixture_ratios.shape[1]):  # Iterate over components
        x = admixture_ratios[:, i]
        smoothness = np.dot(x.T, np.dot(laplacian, x))  # x^T L x
        smoothness_values.append(smoothness)
    
    return np.mean(smoothness_values)


# laplacian eigenmap approach
def compute_continental_admixture_metric_laplacian(ancestry_coords, admixture_ratios):
    # Compute Laplacian
    laplacian = compute_knn_laplacian(ancestry_coords, k=5)

    # Compute average smoothness
    average_smoothness = compute_average_smoothness(laplacian, admixture_ratios)

    return average_smoothness


# Just computes spearman corr between admixture distance and embedding distances
def compute_continental_admixture_metric_dists(ancestry_coords, admixture_ratios, merged_metadata, use_medians=False, only_far=False):
    summarized_df = pd.concat([merged_metadata['Population'],
                               pd.DataFrame(ancestry_coords, index=merged_metadata.index).rename(columns={0: 'emb1', 1: 'emb2'}),
                               pd.DataFrame(admixture_ratios, index=merged_metadata.index)],
                              axis=1)

    if use_medians:
        summarized_df = summarized_df.groupby(['Population']).median().reset_index()

    ancestry_coords2 = summarized_df[['emb1', 'emb2']]
    admixture_ratios2 = summarized_df[np.arange(admixture_ratios.shape[1])]

    ancestry_dists = pdist(ancestry_coords2)
    #admixture_dists = pdist(admixture_ratios)
    k = 5
    while k < 100:
        admixture_dists = compute_geodesic_distances(admixture_ratios2, k=k, metric='euclidean')
        if admixture_dists is None:
            #print('Graph not connected at k={}. Trying k={}'.format(k, k+5))
            k += 5
        else:
            admixture_preservation = preservation_metric(admixture_dists,
                                                         ancestry_dists,
                                                         only_far=only_far)
            return admixture_preservation
    
    print('Graph not connected, even at k=100! Giving up!')
    return None