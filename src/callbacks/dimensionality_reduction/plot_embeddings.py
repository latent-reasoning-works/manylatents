import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scprep

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback
from src.data.hgdp_dataset import HGDPDataset
from src.utils.mappings import make_palette_label_order_HGDP

logger = logging.getLogger(__name__)

class PlotEmbeddings(DimensionalityReductionCallback):
    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        figsize: tuple = (8, 6),
        label_col: str = "Population",
        legend: bool = False,
    ):
        """
        Args:
            save_dir (str): Directory where the plot will be saved.
            experiment_name (str): Name of the experiment to be used in the filename.
            figsize (tuple): Size of the figure (width, height).
            label_col (str): Column in the metadata to use for color-coding.
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.figsize = figsize
        self.label_col = label_col
        self.legend = legend

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"PlotEmbeddings initialized with directory: {self.save_dir} and experiment name: {self.experiment_name}"
        )

    def on_dr_end(self, dataset: any, embeddings: np.ndarray) -> str:
        """
        Plots the first two dimensions of the embeddings and saves the plot as a PNG.
        If the embeddings have more than 2 dimensions, only the first two are used.

        Args:
            dataset (Any): A dataset object with a .metadata attribute and a get_labels() method.
            embeddings (np.ndarray): The computed embeddings (N, D).
            legend (bool): Whether to include a legend in the plot.

        Returns:
            str: The file path to the saved plot.
        """
        if embeddings.shape[1] < 2:
            logger.warning("Not enough dimensions for plotting (need at least 2). Skipping plot.")
            return ""

        # Use only the first two dimensions.
        embeddings_to_plot = embeddings.numpy()[:, :2] if embeddings.shape[1] > 2 else embeddings.numpy()
        logger.info("Using first two dimensions for plotting.")

        labels = None
        # Try to retrieve labels from the dataset.
        if hasattr(dataset, "get_labels") and callable(dataset.get_labels):
            try:
                labels = dataset.get_labels(self.label_col)
            except Exception as e:
                logger.warning(f"Unable to retrieve labels from dataset: {e}")
        # Use metadata from the dataset to get labels.
        # Here, we slice the metadata and embeddings to align if necessary.
        metadata = dataset.metadata[1:]
        labels = metadata['Population'].values
        embeddings_to_plot = embeddings_to_plot[1:]
        
        # Build the palette for HGDP if applicable.
        if isinstance(dataset, HGDPDataset):
            try:
                cmap_pop, _ = make_palette_label_order_HGDP(metadata)
            except Exception as e:
                logger.warning(f"Error building HGDP palette: {e}. Using fallback palette 'viridis'.")
                cmap_pop = 'viridis'
        else:
            logger.info("Dataset is not HGDP. Using default palette 'viridis'.")
            cmap_pop = 'viridis'
        
        figsize = (self.figsize[0], self.figsize[1])
        # Use self.legend in the plotting call.
        if labels is not None:
            scprep.plot.scatter2d(
                embeddings_to_plot, s=8, figsize=figsize,
                cmap=cmap_pop, c=labels,
                ticks=False, legend=self.legend,
                xlabel=' ', ylabel=' ',
                legend_loc='upper center', legend_anchor=(1.0, -0.02), legend_ncol=8,
                label_prefix=None, title='', fontsize=36
            )
        else:
            scprep.plot.scatter2d(
                embeddings_to_plot, s=8, figsize=figsize,
                cmap='viridis', ticks=False, legend=False,
                xlabel=' ', ylabel=' ',
                label_prefix=None, title='', fontsize=36
            )

        plt.xlabel("Dim 1", fontsize=12)
        plt.ylabel("Dim 2", fontsize=12)
        plt.title("Dimensionality Reduction Plot", fontsize=16)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_plot_{self.experiment_name}_{timestamp}.png"
        self.save_path = os.path.join(self.save_dir, filename)

        plt.savefig(self.save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 2D embeddings plot to {self.save_path}")
        return self.save_path
