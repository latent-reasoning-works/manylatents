import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.data.plink_dataset import PlinkDataset

logger = logging.getLogger(__name__)


class PlotAdmixture(EmbeddingCallback):
    """
    Callback to plot embeddings colored by admixture proportions.

    Creates a subplot grid showing each admixture component colored by its proportion,
    using the seismic colormap. Only works with PlinkDataset objects that have admixture
    data loaded.
    """

    def __init__(
        self,
        save_dir: str = "outputs",
        experiment_name: str = "experiment",
        admixture_K: int = 5,
        figsize_per_plot: tuple = (4, 4),
        point_size: float = 4,
        alpha: float = 0.4,
        cmap: str = "seismic",
    ):
        """
        Args:
            save_dir (str): Directory where the plot will be saved.
            experiment_name (str): Name of the experiment for the filename.
            admixture_K (int): Number of admixture components to plot (max 10).
            figsize_per_plot (tuple): Size of each individual subplot (width, height).
            point_size (float): Size of scatter plot points.
            alpha (float): Transparency of points (0 = transparent, 1 = opaque).
            cmap (str): Matplotlib colormap to use for coloring proportions.
        """
        super().__init__()
        if admixture_K > 10:
            raise ValueError("admixture_K must be <= 10")

        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.admixture_K = admixture_K
        self.figsize_per_plot = figsize_per_plot
        self.point_size = point_size
        self.alpha = alpha
        self.cmap = cmap

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"PlotAdmixture initialized with directory: {self.save_dir}, "
            f"experiment name: {self.experiment_name}, K={admixture_K}"
        )

    def _get_subplot_layout(self, K: int) -> tuple:
        """Determine subplot layout based on number of components."""
        if K <= 5:
            return (1, K)
        else:
            # For 6-10, use 2 rows
            ncols = (K + 1) // 2  # Ceiling division
            return (2, ncols)

    def _get_embeddings(self, embeddings: dict) -> np.ndarray:
        """Extract 2D embeddings from embeddings dict."""
        embeddings_data = embeddings["embeddings"]
        if hasattr(embeddings_data, "numpy"):
            emb_np = embeddings_data.numpy()
        else:
            emb_np = embeddings_data
        embeddings_to_plot = emb_np[:, :2] if emb_np.shape[1] > 2 else emb_np
        return embeddings_to_plot

    def _check_admixture_available(self, dataset: any) -> tuple:
        """
        Check if dataset is a PlinkDataset with admixture data loaded.

        Returns:
            tuple: (has_admixture: bool, admixture_df: pd.DataFrame or None)
        """
        if not isinstance(dataset, PlinkDataset):
            logger.warning(
                f"PlotAdmixture skipped: dataset is {type(dataset).__name__}, "
                "not a PlinkDataset"
            )
            return False, None

        if not hasattr(dataset, 'admixture_ratios') or dataset.admixture_ratios is None:
            logger.warning(
                "PlotAdmixture skipped: dataset.admixture_ratios is None. "
                "Make sure admixture data is configured in the dataset."
            )
            return False, None

        # Check if the requested K exists
        K_str = str(self.admixture_K)
        if K_str not in dataset.admixture_ratios:
            logger.warning(
                f"PlotAdmixture skipped: admixture_K={self.admixture_K} not found in dataset. "
                f"Available K values: {list(dataset.admixture_ratios.keys())}"
            )
            return False, None

        admixture_df = dataset.admixture_ratios[K_str]

        # First column is sample ID, so actual number of components is shape[1] - 1
        n_components = admixture_df.shape[1] - 1

        if n_components < self.admixture_K:
            logger.warning(
                f"PlotAdmixture: admixture data has {n_components} components, "
                f"but K={self.admixture_K} was requested. Using {n_components} components."
            )

        return True, admixture_df

    def _plot_admixture(self, embeddings_2d: np.ndarray, admixture_df: pd.DataFrame) -> str:
        """Create admixture proportion subplot grid."""
        # First column is sample ID, skip it and get only numeric admixture columns
        # Admixture file format: sample_id, anc1, anc2, ..., ancK, (population, ancestry_group already removed)
        admixture_numeric = admixture_df.iloc[:, 1:]  # Skip first column (sample ID)

        K = min(self.admixture_K, admixture_numeric.shape[1])
        nrows, ncols = self._get_subplot_layout(K)

        # Calculate total figure size
        fig_width = ncols * self.figsize_per_plot[0]
        fig_height = nrows * self.figsize_per_plot[1]

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

        # Handle case where axes is not an array (single subplot)
        if K == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Plot each admixture component
        for idx in range(K):
            ax = axes[idx]

            # Get admixture proportions for this component
            anc_col = admixture_numeric.iloc[:, idx].values

            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=anc_col,
                s=self.point_size,
                alpha=self.alpha,
                cmap=self.cmap,
                vmin=0,
                vmax=1
            )

            # Remove axis labels, ticks, and tick labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Component {idx+1} proportion', fontsize=13)

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Proportion')

        # Hide unused subplots if K doesn't fill the grid
        for idx in range(K, len(axes)):
            axes[idx].axis('off')

        # Add overall title
        plt.suptitle(
            f'Embeddings: Admixture Proportions (K={K})',
            fontsize=18,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"admixture_plot_K{K}_{self.experiment_name}_{timestamp}.png"
        save_path = os.path.join(self.save_dir, filename)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved admixture plot to {save_path}")
        if wandb.run is not None:
            wandb.log({"admixture_plot": wandb.Image(save_path)})

        return save_path

    def on_latent_end(self, dataset: any, embeddings: dict) -> dict:
        """Main callback entry point."""
        # Check if admixture data is available
        has_admixture, admixture_df = self._check_admixture_available(dataset)

        if not has_admixture:
            # Return empty outputs if we can't plot
            return self.callback_outputs

        # Extract embeddings
        embeddings_2d = self._get_embeddings(embeddings)

        # Create plot
        path = self._plot_admixture(embeddings_2d, admixture_df)

        # Register output
        self.register_output("admixture_plot_path", path)
        return self.callback_outputs
