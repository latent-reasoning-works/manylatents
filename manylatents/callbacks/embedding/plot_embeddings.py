import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

import wandb
from manylatents.callbacks.embedding.base import (
    EmbeddingCallback,
    ColormapInfo,
    ColormapProvider,
)

logger = logging.getLogger(__name__)


class PlotEmbeddings(EmbeddingCallback):
    """
    Callback for plotting 2D embeddings with customizable colormaps and legends.

    This callback creates scatter plots of embeddings, supporting:
    - Categorical coloring with dict-based or ListedColormap colormaps
    - Continuous coloring with colorbars
    - Dataset-provided colormaps via the ColormapProvider protocol
    - WandB integration for logging plots

    The callback can be extended by:
    - Overriding `_get_colormap()` for custom colormap logic
    - Having datasets implement `ColormapProvider.get_colormap_info()`
    """

    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        figsize: tuple = (8, 6),
        label_col: str = "Population",
        legend: bool = False,
        color_by_score: str = None,
        x_label: str = "Dim 1",
        y_label: str = "Dim 2",
        title: str = "Dimensionality Reduction Plot",
        apply_norm: bool = False,
        alpha: float = 0.8,
        log_key: str = "embedding_plot",
        enable_wandb_upload: bool = True,
    ):
        """
        Initialize the PlotEmbeddings callback.

        Args:
            save_dir: Directory where the plot will be saved.
            experiment_name: Name of the experiment for the filename.
            figsize: Figure size (width, height).
            label_col: Column name in metadata to use for labels (fallback).
            legend: Whether to include a legend for categorical data.
            color_by_score: Optional key in embeddings to use for score-based coloring.
                           Special value "tangent_space" enables categorical coloring.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            title: Title for the plot.
            apply_norm: Whether to apply dynamic normalization to continuous color arrays.
            alpha: Transparency of the points (0 = transparent, 1 = opaque).
            log_key: Key for WandB logging. Can be customized for step-aware logging.
            enable_wandb_upload: Whether to upload to WandB when available.
        """
        super().__init__()
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
        self.log_key = log_key
        self.enable_wandb_upload = enable_wandb_upload

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"PlotEmbeddings initialized with directory: {self.save_dir} "
            f"and experiment name: {self.experiment_name}"
        )
        if not enable_wandb_upload:
            logger.info("WandB upload disabled - running in offline mode")

    def _get_colormap(self, dataset: Any) -> ColormapInfo:
        """
        Get colormap information for plotting.

        Resolution order:
        1. If color_by_score is "tangent_space", return categorical colormap.
        2. If color_by_score is set (continuous), return "viridis".
        3. If dataset implements ColormapProvider protocol, use its colormap.
        4. Fall back to "viridis".

        This method can be overridden in subclasses for custom colormap logic,
        particularly useful for downstream packages like manylatents-omics.

        Args:
            dataset: The dataset object, optionally implementing ColormapProvider.

        Returns:
            ColormapInfo containing the colormap and optional label names.
        """
        # Special case: tangent_space categorical coloring
        if self.color_by_score == "tangent_space":
            return ColormapInfo(
                cmap=ListedColormap(["#1f77b4", "#ff7f0e"]),
                label_names={0: "Dim ~ 1", 1: "Dim ~ 2"},
                is_categorical=True,
            )

        # Continuous score coloring
        if self.color_by_score is not None:
            return ColormapInfo(cmap="viridis", label_names=None, is_categorical=False)

        # Dataset-provided colormap via protocol (no isinstance checks for specific types)
        if isinstance(dataset, ColormapProvider):
            return dataset.get_colormap_info()

        # Fallback
        return ColormapInfo(cmap="viridis", label_names=None, is_categorical=True)

    def _get_embeddings(self, embeddings: dict) -> np.ndarray:
        """
        Extract and prepare embeddings for 2D plotting.

        Args:
            embeddings: Dictionary containing 'embeddings' key.

        Returns:
            2D numpy array of shape (n_samples, 2) for plotting.
        """
        emb = embeddings["embeddings"]

        # Convert torch tensors to numpy
        if hasattr(emb, "numpy"):
            emb_np = emb.numpy()
        elif hasattr(emb, "cpu"):  # Handle GPU tensors
            emb_np = emb.cpu().numpy()
        else:
            emb_np = np.asarray(emb)

        # Take first 2 dimensions for plotting
        embeddings_to_plot = emb_np[:, :2] if emb_np.shape[1] > 2 else emb_np

        return embeddings_to_plot

    def _get_color_array(self, dataset: Any, embeddings: dict) -> Optional[np.ndarray]:
        """
        Extract color array for scatter plot coloring.

        Resolution order:
        1. If color_by_score is set, look in embeddings['scores'][key] or embeddings[key].
        2. Use embeddings['label'] if available.
        3. Fall back to dataset.get_labels().

        Args:
            dataset: Dataset object with optional get_labels() method.
            embeddings: Dictionary potentially containing 'label' or 'scores'.

        Returns:
            Numpy array of colors/labels, or None if unavailable.
        """
        color_array = None

        if self.color_by_score is not None:
            # Try to get from scores dict first
            scores = embeddings.get("scores")
            if scores is not None and isinstance(scores, dict):
                color_array = scores.get(self.color_by_score)

            # Fall back to direct key
            if color_array is None:
                color_array = embeddings.get(self.color_by_score)

            if color_array is None:
                logger.warning(
                    f"Coloring key '{self.color_by_score}' not found; "
                    "falling back to label-based coloring."
                )
            else:
                logger.info(f"Using '{self.color_by_score}' for coloring.")

        # Label-based coloring fallback
        if color_array is None:
            if "label" in embeddings and embeddings["label"] is not None:
                color_array = embeddings["label"]
            elif hasattr(dataset, "get_labels"):
                # Note: get_labels() doesn't take label_col argument in most implementations
                try:
                    color_array = dataset.get_labels(self.label_col)
                except TypeError:
                    # Fall back to no-argument version
                    color_array = dataset.get_labels()

        # Convert to numpy array
        if color_array is not None:
            if hasattr(color_array, "numpy"):
                color_array = color_array.numpy()
            elif hasattr(color_array, "cpu"):
                color_array = color_array.cpu().numpy()
            color_array = np.asarray(color_array)
            return color_array

        return None

    def _apply_dict_colormap(
        self, labels: np.ndarray, cmap_dict: Dict[Union[int, str], str]
    ) -> List[str]:
        """
        Apply a dict-based colormap to label array.

        Args:
            labels: Array of label values (int or str).
            cmap_dict: Mapping from label values to color strings.

        Returns:
            List of color strings for each point.
        """
        default_color = "#808080"  # Gray for unknown labels
        return [cmap_dict.get(label, default_color) for label in labels]

    def _add_categorical_legend(self, ax: plt.Axes, cmap_info: ColormapInfo) -> None:
        """
        Add categorical legend using ColormapInfo label_names.

        Args:
            ax: Matplotlib axes object.
            cmap_info: ColormapInfo with label_names mapping.
        """
        if cmap_info.label_names is None:
            return

        patches = []
        for idx, name in sorted(cmap_info.label_names.items()):
            if isinstance(cmap_info.cmap, dict):
                color = cmap_info.cmap.get(idx, "#808080")
            elif isinstance(cmap_info.cmap, ListedColormap):
                colors = cmap_info.cmap.colors
                color = colors[idx % len(colors)]
            else:
                # String colormap name - get color from matplotlib
                cmap = plt.colormaps.get_cmap(cmap_info.cmap)
                max_idx = max(cmap_info.label_names.keys())
                color = cmap(idx / max_idx) if max_idx > 0 else cmap(0)
            patches.append(Patch(facecolor=color, label=name))

        ax.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(8, len(patches)),
            fontsize=8,
        )

    def _add_categorical_legend_from_data(
        self,
        ax: plt.Axes,
        labels: np.ndarray,
        cmap_info: ColormapInfo,
    ) -> None:
        """
        Add categorical legend when no label_names provided, using unique labels from data.

        Args:
            ax: Matplotlib axes object.
            labels: Array of label values.
            cmap_info: ColormapInfo containing the colormap.
        """
        unique_labels = sorted(set(labels))

        patches = []
        for lbl in unique_labels:
            if isinstance(cmap_info.cmap, dict):
                color = cmap_info.cmap.get(lbl, "#808080")
            elif isinstance(cmap_info.cmap, ListedColormap):
                colors = cmap_info.cmap.colors
                # For numeric labels, use as index; for others, use hash
                idx = lbl if isinstance(lbl, (int, np.integer)) else hash(lbl)
                color = colors[int(idx) % len(colors)]
            else:
                cmap = plt.colormaps.get_cmap(cmap_info.cmap)
                n_unique = len(unique_labels)
                idx = unique_labels.index(lbl)
                color = cmap(idx / max(1, n_unique - 1))
            patches.append(Patch(facecolor=color, label=str(lbl)))

        ax.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(8, len(patches)),
            fontsize=8,
        )

    def _upload_to_wandb(self) -> None:
        """Upload plot to WandB if enabled and run is active."""
        if self.enable_wandb_upload and wandb.run is not None:
            wandb.log({self.log_key: wandb.Image(self.save_path)})
            logger.info(f"Uploaded plot to WandB with key: {self.log_key}")
        elif not self.enable_wandb_upload:
            logger.info("WandB upload skipped (offline mode)")
        else:
            logger.info("WandB upload skipped (no active run)")

    def _plot_embeddings(
        self,
        dataset: Any,
        embeddings_to_plot: np.ndarray,
        color_array: Optional[np.ndarray],
    ) -> str:
        """
        Create and save the embedding scatter plot using matplotlib.

        Args:
            dataset: Dataset for colormap information.
            embeddings_to_plot: 2D array of shape (n_samples, 2).
            color_array: Array of colors/labels for each point.

        Returns:
            Path to the saved plot file.
        """
        cmap_info = self._get_colormap(dataset)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Handle different colormap types
        if cmap_info.is_categorical and isinstance(cmap_info.cmap, dict):
            # Dict-based colormap: map labels to colors directly
            if color_array is not None:
                colors = self._apply_dict_colormap(color_array, cmap_info.cmap)
                scatter = ax.scatter(
                    embeddings_to_plot[:, 0],
                    embeddings_to_plot[:, 1],
                    c=colors,
                    s=8,
                    alpha=self.alpha,
                )
            else:
                scatter = ax.scatter(
                    embeddings_to_plot[:, 0],
                    embeddings_to_plot[:, 1],
                    s=8,
                    alpha=self.alpha,
                )

            # Create categorical legend
            if self.legend:
                if cmap_info.label_names:
                    self._add_categorical_legend(ax, cmap_info)
                elif color_array is not None:
                    self._add_categorical_legend_from_data(ax, color_array, cmap_info)

        elif isinstance(cmap_info.cmap, ListedColormap):
            # ListedColormap for discrete categories
            scatter = ax.scatter(
                embeddings_to_plot[:, 0],
                embeddings_to_plot[:, 1],
                c=color_array,
                cmap=cmap_info.cmap,
                s=8,
                alpha=self.alpha,
            )

            if self.legend and cmap_info.label_names:
                self._add_categorical_legend(ax, cmap_info)

        else:
            # String colormap name (continuous or categorical fallback)
            norm = None
            if (
                not cmap_info.is_categorical
                and self.apply_norm
                and color_array is not None
            ):
                norm = Normalize(vmin=color_array.min(), vmax=color_array.max())

            scatter = ax.scatter(
                embeddings_to_plot[:, 0],
                embeddings_to_plot[:, 1],
                c=color_array,
                cmap=cmap_info.cmap,
                norm=norm,
                s=8,
                alpha=self.alpha,
            )

            # Add colorbar for continuous data
            if not cmap_info.is_categorical and self.color_by_score:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f"{self.color_by_score} Score")
            elif self.legend and cmap_info.is_categorical and color_array is not None:
                # Categorical with string colormap
                self._add_categorical_legend_from_data(ax, color_array, cmap_info)

        # Configure axes
        ax.set_xlabel(self.x_label, fontsize=12)
        ax.set_ylabel(self.y_label, fontsize=12)
        ax.set_title(self.title, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"embedding_plot_{self.experiment_name}_{timestamp}.png"
        self.save_path = os.path.join(self.save_dir, filename)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        logger.info(f"Saved 2D embeddings plot to {self.save_path}")

        # WandB upload
        self._upload_to_wandb()

        return self.save_path

    def on_latent_end(self, dataset: Any, embeddings: dict) -> dict:
        """
        Called when the latent process is complete.

        Creates a 2D scatter plot of the embeddings and optionally uploads to WandB.

        Args:
            dataset: A dataset object that may implement ColormapProvider.
            embeddings: Dictionary containing 'embeddings' and optionally 'label', 'scores'.

        Returns:
            Dictionary containing callback outputs including 'embedding_plot_path'.
        """
        emb2d = self._get_embeddings(embeddings)
        colors = self._get_color_array(dataset, embeddings)
        path = self._plot_embeddings(dataset, emb2d, colors)
        self.register_output("embedding_plot_path", path)
        return self.callback_outputs
