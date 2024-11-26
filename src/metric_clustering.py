import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import adjusted_rand_score
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

def sweep_HDBSCAN(data, true_labels, param_grid, **kwargs):
    """
    Sweep HDBSCAN parameters using a grid and calculate ARI.

    Parameters
    ----------
    - data: array-like, shape (n_samples, n_features)
    - true_labels: array-like, shape (n_samples,)
    - param_grid: dict
        Dictionary of parameter lists to sweep, e.g.,{'min_cluster_size': [10, 20], 'min_samples': [5, 10]}.
    - **kwargs: Additional parameters for HDBSCAN.

    Returns
    ----------
    - best_result: dict
        Contains the best parameter combination, the highest ARI score, and the corresponding cluster labels.
        TO reproduce the best: 
            best_params = best_result["best_params"]
            clusterer = hdbscan.HDBSCAN(**best_params)
    - average_result: dict
        Contains the mean and standard deviation of ARI scores across all parameter combinations.
    """
    best_ari = -1
    best_params = None
    best_clusters = None
    ari_scores = []
    param_ari_scores = {}
    # Generate all combinations of parameters from the grid
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params in param_combinations:
        # Merge user-defined parameters with swept parameters
        clusterer_params = {**kwargs, **params}
        
        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(**clusterer_params)
        cluster_labels = clusterer.fit_predict(data)

        # Calculate ARI if clustering was successful
        if len(set(cluster_labels)) > 1:  # Ensure at least two clusters are formed
            ari = adjusted_rand_score(true_labels, cluster_labels)
            ari_scores.append(ari)
            param_ari_scores[tuple(params.items())] = ari

            # Update best ARI and parameters
            if ari > best_ari:
                best_ari = ari
                best_params = params
                best_clusters = cluster_labels

    # Calculate statistics
    mean_ari = np.mean(ari_scores) if ari_scores else 0.0
    std_ari = np.std(ari_scores) if ari_scores else 0.0

    best_result = {
        "best_params": best_params,
        "best_ari": best_ari,
        "best_clusters": best_clusters,
    }

    average_result = {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "all_ari_scores": param_ari_scores,
    }

    return best_result, average_result

def make_confusion_heatmap(true_labels, clusterer_labels, ax=None, colorbar=True, method = 'HDBSCAN'):
    """
    Create a confusion heatmap of the populations and clusters.

    Parameters
    ----------
    true_labels : array-like
        The true population labels. 
    clusterer_labels : array-like
        The cluster labels assigned by the clustering algorithm.
    ax : matplotlib axis
    colorbar : bool, optional, default: True
    method : str, optional, default: 'HDBSCAN', can also be 'PHATE'

    Returns 
    -------
    heatmap : seaborn heatmap
        The heatmap of the confusion matrix.
    """
    df = pd.DataFrame(columns=['Cluster', 'Population'])
    df['Cluster'] = clusterer_labels
    df['Population'] = true_labels

    # Filter out the -1 cluster (unclustered data)
    df = df[df['Cluster'] != -1]
    # Create a cross-tabulation of clusters and populations
    crosstab = pd.crosstab(df['Cluster'], df['Population'])
    # Convert counts to proportions
    proportions = crosstab.div(crosstab.sum(axis=1), axis=0)

    # Reorder the populations to match the clusters
    ordered_populations = proportions.idxmax().sort_values().index
    proportions = proportions[ordered_populations]

    # Plot the heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(proportions, cmap='viridis', annot=False,linewidths=0.05, linecolor='purple', 
                            cbar=colorbar if colorbar else None, cbar_kws={'shrink': 0.8} if colorbar else None, ax=ax)
    heatmap.invert_yaxis()
    ax.set_xlabel('Population')
    ax.set_ylabel(method + ' Cluster')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12)

    # Adjust the color bar tick labels font size
    if colorbar:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
    return heatmap

