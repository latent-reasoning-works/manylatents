import matplotlib.pyplot as plt
import scprep
import seaborn as sns
import pandas as pd

# Generate PHATE plots
def plot_phate_results(phate_embs_list, 
                       metadata, 
                       ts, 
                       param_values, 
                       param_name, 
                       cmap, 
                       output_dir):
    num_vals = len(phate_embs_list)
    num_t = len(phate_embs_list[0])

    # Ensure axes are always treated as 2D arrays
    fig, ax = plt.subplots(
        figsize=(10 * num_t, 10 * num_vals),
        nrows=num_vals,
        ncols=num_t,
        gridspec_kw={'wspace': 0.08},
        squeeze=False  # Ensure ax is always 2D
    )

    for i, embs in enumerate(phate_embs_list):
        for j, emb in enumerate(embs):
            scprep.plot.scatter2d(
                emb,
                s=5,
                ax=ax[i, j],
                c=metadata['Population'].values,
                cmap=cmap,
                xticks=False,
                yticks=False,
                legend=False,
                legend_loc='lower center',
                legend_anchor=(0.5, -0.35),
                legend_ncol=8,
                label_prefix="PHATE ",
                fontsize=8
            )
            ax[i, j].set_title('t={} {}={}'.format(ts[j], param_name, param_values[i]), fontsize=30)
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()

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
