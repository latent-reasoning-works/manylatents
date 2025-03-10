import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        label_col: str = "Population"
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

        Returns:
            str: The file path to the saved plot.
        """
        if embeddings.shape[1] < 2:
            logger.warning("Not enough dimensions for plotting (need at least 2). Skipping plot.")
            return ""

        # Use only the first two dimensions.
        embeddings_to_plot = embeddings[:, :2] if embeddings.shape[1] > 2 else embeddings
        logger.info("Using first two dimensions for plotting.")

        labels = None
        palette = None

        # Try to retrieve labels from the dataset.
        if hasattr(dataset, "get_labels") and callable(dataset.get_labels):
            try:
                labels = dataset.get_labels(self.label_col)
            except Exception as e:
                logger.warning(f"Unable to retrieve labels from dataset: {e}")

        if isinstance(dataset, HGDPDataset):
            try:
                # Extract populations from the metadata using the specified label column.
                populations = dataset.metadata[self.label_col].astype(str).values
                # Attempt to extract superpopulations; fall back to populations if not available.
                if "Superpopulation" in dataset.metadata.columns:
                    superpopulations = dataset.metadata["Superpopulation"].values
                else:
                    logger.warning(
                        "Superpopulation column not found in metadata; using population labels as fallback."
                    )
                    superpopulations = populations

                # Call the mapping function with both arrays.
                coarse_dict, fine_dict, _, _ = make_palette_label_order_HGDP(populations, superpopulations)
                palette = fine_dict
                if labels is not None:
                    unique_labels = np.unique(labels)
                    # Subset the palette to only include labels present.
                    palette = {lab: palette.get(lab, "gray") for lab in unique_labels}
            except Exception as e:
                logger.warning(f"Error building HGDP palette: {e}")
        else:
            logger.info("Dataset is not HGDP. Using default palette 'viridis' if labels are provided.")

        # Plotting
        plt.figure(figsize=self.figsize)
        if labels is not None:
            sns.scatterplot(
                x=embeddings_to_plot[:, 0],
                y=embeddings_to_plot[:, 1],
                hue=labels,
                palette=palette if palette else "viridis",
                alpha=0.7
            )
            plt.legend()
        else:
            plt.scatter(embeddings_to_plot[:, 0], embeddings_to_plot[:, 1], alpha=0.7)

        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title("Dimensionality Reduction Plot")

        # Generate the filename following the convention: embedding_plot_{experiment_name}_{timestamp}.png
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_plot_{self.experiment_name}_{timestamp}.png"
        self.save_path = os.path.join(self.save_dir, filename)

        plt.savefig(self.save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 2D embeddings plot to {self.save_path}")
        return self.save_path
