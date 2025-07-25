import logging
import os
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scprep
from matplotlib.colors import ListedColormap

import wandb
from src.callbacks.embedding.base import EmbeddingCallback
from src.data.hgdp_dataset import HGDPDataset
from src.data.ukbb_dataset import UKBBDataset
from src.data.mhi_dataset import MHIDataset
from src.data.aou_dataset import AOUDataset
from src.utils.mappings import cmap_pop as cmap_pop_HGDP
from src.utils.mappings import cmap_ukbb_superpops as cmap_pop_UKBB
from src.utils.mappings import cmap_mhi_superpops as cmap_pop_MHI
from src.utils.mappings import race_ethnicity_only_pca_colors as cmap_pop_AOU

logger = logging.getLogger(__name__)

class PlotEmbeddings(EmbeddingCallback):
    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        figsize: tuple = (8, 6),
        label_col: str = "Population",
        legend: bool = False,
        color_by_score: str = None,  # e.g. "tangent_space" or any metric key
        x_label: str = "Dim 1",
        y_label: str = "Dim 2",
        title: str = "Dimensionality Reduction Plot",
        apply_norm: bool = False,    ## EXPERIMENTAL, NOT INCONSISTENT IN TSA CASE
        alpha: float = 0.8          
    ):
        super().__init__()
        """
        Args:
            save_dir (str): Directory where the plot will be saved.
            experiment_name (str): Name of the experiment for the filename.
            figsize (tuple): Figure size (width, height).
            label_col (str): Column name in metadata to use for labels (fallback).
            legend (bool): Whether to include a legend.
            color_by_score (str): Optional key in embeddings to use for continuous score-based coloring.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title for the plot.
            apply_norm (bool): Whether to apply dynamic normalization to the color array.
            alpha (float): Transparency of the points (0 = completely transparent, 1 = opaque).
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.figsize = figsize
        self.label_col = label_col
        self.legend = legend
        self.color_by_score = color_by_score
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.apply_norm = apply_norm
        self.alpha = alpha

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"PlotEmbeddings initialized with directory: {self.save_dir} and experiment name: {self.experiment_name}"
        )

    def _get_colormap(self, dataset: any) -> any:
        if self.color_by_score == "tangent_space":
            # Define discrete colors for the two categories (adjust colors as needed)
            return ListedColormap(["#1f77b4", "#ff7f0e"])
        if self.color_by_score is not None:
            return "viridis"
        if isinstance(dataset, HGDPDataset):
            cmap = cmap_pop_HGDP
        elif isinstance(dataset, UKBBDataset):
            cmap = cmap_pop_UKBB
        elif isinstance(dataset, MHIDataset):
            cmap = cmap_pop_MHI
        elif isinstance(dataset, AOUDataset):
            cmap = cmap_pop_AOU
        else:
            cmap = "viridis"
        return cmap

    def _get_embeddings(self, embeddings: dict) -> np.ndarray:
        embeddings = embeddings["embeddings"]
        if hasattr(embeddings, "numpy"):
            emb_np = embeddings.numpy()
        else:
            emb_np = embeddings
        embeddings_to_plot = emb_np[:, :2] if emb_np.shape[1] > 2 else emb_np
        return embeddings_to_plot[1:]

    def _get_color_array(self, dataset: any, embeddings: dict) -> np.ndarray:
        color_array = None
        if self.color_by_score is not None:
            scores = embeddings.get("scores")
            if scores is not None and isinstance(scores, dict):
                color_array = scores.get(self.color_by_score)
            else:
                color_array = embeddings.get(self.color_by_score)
            if color_array is None:
                logger.warning(
                    f"Coloring key '{self.color_by_score}' not found in embeddings; falling back to label-based coloring."
                )
                if "label" in embeddings and embeddings["label"] is not None:
                    color_array = embeddings["label"]
                else:
                    color_array = dataset.get_labels(self.label_col)
            else:
                logger.info(f"Using '{self.color_by_score}' for coloring.")
        else:
            if "label" in embeddings and embeddings["label"] is not None:
                color_array = embeddings["label"]
            else:
                color_array = dataset.get_labels(self.label_col)
        if color_array is not None:
            if hasattr(color_array, "numpy"):
                color_array = color_array.numpy()
            color_array = np.asarray(color_array)
            return color_array[1:]
        return None
    
    def _plot_embeddings(self, dataset: any, embeddings_to_plot: np.ndarray, color_array: np.ndarray) -> str:
        cmap = self._get_colormap(dataset)
        fig_size = (self.figsize[0], self.figsize[1])
        
        norm = None
        # For continuous scores, you may want to normalize. For tangent_space (categorical), skip normalization.
        if self.color_by_score is not None and self.color_by_score != "tangent_space" and color_array is not None and self.apply_norm:
            score_min = float(color_array.min())
            score_max = float(color_array.max())
            norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
        ax = scprep.plot.scatter2d(
            embeddings_to_plot,
            s=8,
            figsize=fig_size,
            cmap=cmap,
            c=color_array,
            ticks=False,
            legend=self.legend,
            xlabel=' ',
            ylabel=' ',
            legend_loc='upper center',
            legend_anchor=(1.0, -0.02),
            legend_ncol=8,
            label_prefix=None,
            title='',
            fontsize=36,
            alpha=self.alpha
        )
        
        if norm is not None and ax.collections:
            mappable = ax.collections[0]
            mappable.set_norm(norm)
            mappable.set_array(np.asarray(color_array))
        
        plt.xlabel(self.x_label, fontsize=12)
        plt.ylabel(self.y_label, fontsize=12)
        plt.title(self.title, fontsize=16)
        
        # For tangent_space, create a categorical legend
        if self.color_by_score == "tangent_space":
            from matplotlib.patches import Patch
            # Adjust the colors to match those returned by _get_colormap (ListedColormap)
            legend_elements = [
                Patch(facecolor="#1f77b4", label="Dim ~ 1"),
                Patch(facecolor="#ff7f0e", label="Dim ~ 2")
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        elif self.color_by_score is not None and ax.collections:
            # For continuous scores, create a colorbar
            mappable = ax.collections[0]
            cbar = plt.colorbar(mappable)
            cbar.set_label(f"{self.color_by_score} Score")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_plot_{self.experiment_name}_{timestamp}.png"
        self.save_path = os.path.join(self.save_dir, filename)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved 2D embeddings plot to {self.save_path}")
        if wandb.run is not None:
            wandb.log({"embedding_plot": wandb.Image(self.save_path)})
        return self.save_path

    def on_latent_end(self, dataset: any, embeddings: dict) -> str:
        emb2d = self._get_embeddings(embeddings)
        colors = self._get_color_array(dataset, embeddings)
        path = self._plot_embeddings(dataset, emb2d, colors)
        self.register_output("embedding_plot_path", path)
        return self.callback_outputs
