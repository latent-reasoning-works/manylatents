import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from typing import List, Dict

sns.set_context("notebook", font_scale=0.5)  # Adjust font_scale to make text smaller as needed

def plot_embeddings(data: np.ndarray, 
                    labels: List[str], 
                    title: str, 
                    color_dict: Dict[str, str], 
                    label_order: List[str], 
                    label_positions: bool = True, 
                    ax: Axes | None = None, **kwargs):
    """Plot 2D embeddings with annotations and specific color mapping.

    Args:
        data (np.ndarray): 2D coordinates of the embeddings.
        labels (List[str]): List of labels for each data point.
        title (str): Title of the plot.
        color_dict (Dict[str, str]): Dictionary mapping labels to colors.
        label_positions (bool): Whether to annotate average positions for each label.

    Kwargs:
        Additional keyword arguments for sns.scatterplot (e.g., legend positioning).
    """
    unique_labels = np.unique(labels)
    palette = {label: color_dict[label] for label in unique_labels if label in color_dict}
    
    if not ax:
        create_new_plot = True
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        create_new_plot = False

    sns.scatterplot(x=data[:, 0], 
                    y=data[:, 1], 
                    hue=labels, 
                    palette=palette, 
                    ax=ax, 
                    hue_order=label_order,
                    **kwargs)

    if create_new_plot:
        #ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    else:
        # we dont need ticks or ticklabels
        ax.get_legend().remove()
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

    if label_positions:
        for label in unique_labels:
            indices = np.where(np.array(labels) == label)
            mean_position = np.mean(data[indices], axis=0)
            ax.text(mean_position[0], 
                    mean_position[1], 
                    label, 
                    fontsize=5, 
                    weight='bold', 
                    ha='center')

    if create_new_plot:
        fig.tight_layout()

        # Dont plot title if none (empty string) supplied
        if title:
            plt.savefig(f"{title.replace(' ', '_').lower()}.jpg", 
                        format='jpg', 
                        dpi=300)

            plt.close()

def generate_filenames(**kwargs):
    """
    Generate a unique filename based on the dataset type, manifold algorithm, 
    and relevant parameters using keyword arguments.

    Args:
        **kwargs: Arbitrary keyword arguments.

    Returns:
        str: A unique filename or identifier for the run configuration.
    """
    base_name = f"{kwargs.get('dataset_type')}_{kwargs.get('manifold_algo')}"

    if kwargs.get('manifold_algo') == "pca":
        base_name += f"_comp{kwargs.get('components')}"
        model_name, figure_name = base_name, base_name
    elif kwargs.get('manifold_algo') == "tsne":
        base_name += f"_comp{kwargs.get('components')}_perplex{kwargs.get('perplexity')}_ee{kwargs.get('early_exaggeration')}"
        model_name, figure_name = base_name, base_name
    elif kwargs.get('manifold_algo') == "phate":
        base_name += f"_knn{kwargs.get('knn')}_decay{kwargs.get('decay')}"
        model_name = base_name # only depends on knn and decay parameters
        base_name += f"_gamma{kwargs.get('gamma')}_t{kwargs.get('t')}"
        figure_name = base_name # add in gamma and t

    if kwargs.get('label_positions', False):
        figure_name += "_labelpos"

    return model_name, figure_name

