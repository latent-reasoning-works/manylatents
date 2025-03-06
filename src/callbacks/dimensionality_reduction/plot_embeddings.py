# File: src/callbacks/dimensionality_reduction/plot_embeddings.py

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.callbacks.dimensionality_reduction.base import DimensionalityReductionCallback

logger = logging.getLogger(__name__)

class PlotEmbeddings(DimensionalityReductionCallback):
    def __init__(
        self,
        save_dir: str = "outputs",
        filename: str = "embedding_plot.png",
        figsize: tuple = (8, 6)
    ):
        """
        Args:
            save_dir (str): Directory where the plot will be saved.
            filename (str): Name of the output plot file.
            figsize (tuple): Size of the figure (width, height).
        """
        self.save_dir = save_dir
        self.filename = filename
        self.figsize = figsize

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"PlotEmbeddings initialized with directory: {self.save_dir} and filename: {self.filename}")

    def on_dr_end(self, original: np.ndarray, embeddings: np.ndarray, labels: np.ndarray = None) -> None:
        """
        Plots the 2D embeddings (e.g. PC1 vs PC2) and saves them as a PNG.

        Args:
            original (np.ndarray): The original data (ignored here, but part of the DR callback signature).
            embeddings (np.ndarray): The computed embeddings of shape (N, 2).
            labels (np.ndarray, optional): Optional labels for color-coding in the scatter plot.
        """
        if embeddings.shape[1] != 2:
            logger.warning("Skipping plot: Embeddings must be 2D for visualization.")
            return

        # Create figure and axes
        plt.figure(figsize=self.figsize)

        if labels is not None:
            sns.scatterplot(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                hue=labels,
                palette="viridis",
                alpha=0.7
            )
            plt.legend()
        else:
            plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7)

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Dimensionality Reduction Plot")

        # Build full path with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(self.filename)
        out_filename = f"{base}_{timestamp}{ext}"
        save_path = os.path.join(self.save_dir, out_filename)

        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved 2D embeddings plot to {save_path}")