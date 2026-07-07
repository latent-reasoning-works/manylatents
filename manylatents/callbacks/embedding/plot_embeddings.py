import json
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

try:
    import wandb
    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None
from manylatents.callbacks.embedding.base import (
    EmbeddingCallback,
    ColormapInfo,
    ColormapProvider,
)
from manylatents.utils.plotting import resolve_colormap, merge_colormap_info

logger = logging.getLogger(__name__)

_MAX_CATEGORICAL_UNIQUE = 50  # auto-detect threshold for categorical vs continuous


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
        color_by: str = None,
        color_by_score: str = None,
        x_label: str = "Dim 1",
        y_label: str = "Dim 2",
        title: str = "Dimensionality Reduction Plot",
        apply_norm: bool = False,
        alpha: float = 0.8,
        log_key: str = "embedding_plot",
        enable_wandb_upload: bool = True,
        export_plot_data: bool = True,
        # User overrides for colormap (take precedence over metric/dataset)
        cmap_override: str = None,
        is_categorical_override: bool = None,
        label_format_override: str = None,
    ):
        """
        Initialize the PlotEmbeddings callback.

        Args:
            save_dir: Directory where the plot will be saved.
            experiment_name: Name of the experiment for the filename.
            figsize: Figure size (width, height).
            label_col: Column name in metadata to use for labels (fallback).
            legend: Whether to include a legend for categorical data.
            color_by: Key in the LatentOutputs dict to use for coloring.
                     Special values:
                       - "embeddings": color by first component of the embedding
                         (e.g. cluster labels from Leiden, PC1 from PCA).
                         Auto-detects categorical vs continuous.
                       - Any other string: looked up in scores, then top-level keys.
                       - None (default): use embeddings["label"] or dataset.get_labels().
            color_by_score: Deprecated alias for color_by. Use color_by instead.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            title: Title for the plot.
            apply_norm: Whether to apply dynamic normalization to continuous color arrays.
            alpha: Transparency of the points (0 = transparent, 1 = opaque).
            log_key: Key for WandB logging. Can be customized for step-aware logging.
            enable_wandb_upload: Whether to upload to WandB when available.
            export_plot_data: Whether to also write a machine-readable sidecar
                (coords + per-point color values + resolved colormap) next to the
                PNG, so downstream tools can re-color the embedding interactively.
            cmap_override: Override colormap (e.g., "viridis", "plasma"). Takes precedence
                          over metric-declared and dataset-declared colormaps.
            is_categorical_override: Override categorical/continuous treatment.
            label_format_override: Override label format string (e.g., "Dim = {}").
        """
        super().__init__()
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.figsize = figsize
        self.label_col = label_col
        self.legend = legend

        # Resolve color_by vs deprecated color_by_score
        if color_by is not None:
            self.color_by = color_by
        elif color_by_score is not None:
            warnings.warn(
                "color_by_score is deprecated, use color_by instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.color_by = color_by_score
        else:
            self.color_by = None

        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.apply_norm = apply_norm
        self.alpha = alpha
        self.log_key = log_key
        self.enable_wandb_upload = enable_wandb_upload
        self.export_plot_data = export_plot_data
        # User overrides
        self.cmap_override = cmap_override
        self.is_categorical_override = is_categorical_override
        self.label_format_override = label_format_override

        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(
            f"PlotEmbeddings initialized with directory: {self.save_dir} "
            f"and experiment name: {self.experiment_name}"
        )
        if not enable_wandb_upload:
            logger.info("WandB upload disabled - running in offline mode")

    def _get_colormap(
        self, dataset: Any, embeddings: Optional[dict] = None
    ) -> ColormapInfo:
        """
        Get colormap information for plotting.

        Resolution order (highest to lowest priority):
        1. User overrides (cmap_override, is_categorical_override, label_format_override)
        2. Metric-declared via scores["<color_by_score>__viz"]
        3. Dataset-provided via ColormapProvider protocol
        4. Inferred defaults (viridis)

        This method can be overridden in subclasses for custom colormap logic,
        particularly useful for downstream packages like manylatents-omics.

        Args:
            dataset: The dataset object, optionally implementing ColormapProvider.
            embeddings: The embeddings dict, used to look up metric viz metadata.

        Returns:
            ColormapInfo containing the colormap and optional label names.
        """
        base_info = None

        # Check for metric-declared viz metadata in scores
        if self.color_by is not None and self.color_by != "embeddings" and embeddings is not None:
            scores = embeddings.get("scores", {})
            viz_key = f"{self.color_by}__viz"
            if viz_key in scores and isinstance(scores[viz_key], ColormapInfo):
                base_info = scores[viz_key]
                logger.debug(f"Using metric-declared colormap from {viz_key}")

        # Fall back to dataset-provided colormap
        if base_info is None and isinstance(dataset, ColormapProvider):
            base_info = dataset.get_colormap_info()
            logger.debug("Using dataset-provided colormap via ColormapProvider")

        # Fall back to defaults based on color_by mode
        if base_info is None:
            if self.color_by == "embeddings":
                # Auto-detected later in _get_color_array; default to categorical
                # (overridden by _infer_categorical if needed)
                base_info = ColormapInfo(
                    cmap="viridis", label_names=None, is_categorical=True
                )
            elif self.color_by is not None:
                # Default to continuous for score coloring (metric can override via __viz)
                base_info = ColormapInfo(
                    cmap="viridis", label_names=None, is_categorical=False
                )
            else:
                # Default to categorical for label coloring
                base_info = ColormapInfo(
                    cmap="viridis", label_names=None, is_categorical=True
                )

        # Apply user overrides
        return merge_colormap_info(
            base=base_info,
            cmap_override=self.cmap_override,
            is_categorical_override=self.is_categorical_override,
            label_format_override=self.label_format_override,
        )

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

        # Ensure 2D for plotting: take first 2 dims, or pad if only 1
        if emb_np.shape[1] > 2:
            embeddings_to_plot = emb_np[:, :2]
        elif emb_np.shape[1] == 1:
            embeddings_to_plot = np.column_stack([emb_np[:, 0], np.zeros(len(emb_np))])
        else:
            embeddings_to_plot = emb_np

        return embeddings_to_plot

    @staticmethod
    def _is_categorical(values: np.ndarray) -> bool:
        """Auto-detect whether a 1-D array should be treated as categorical.

        Returns True when every value is integer-valued (even if stored as
        float) **and** the number of unique values is at most
        ``_MAX_CATEGORICAL_UNIQUE``.
        """
        if not np.issubdtype(values.dtype, np.number):
            return True  # string / object → always categorical
        if np.issubdtype(values.dtype, np.integer):
            return len(np.unique(values)) <= _MAX_CATEGORICAL_UNIQUE
        # Float dtype: check if all values are whole numbers
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return False
        all_integer = np.all(np.equal(np.mod(finite, 1), 0))
        return all_integer and len(np.unique(finite)) <= _MAX_CATEGORICAL_UNIQUE

    def _get_color_array(self, dataset: Any, embeddings: dict) -> Optional[np.ndarray]:
        """
        Extract color array for scatter plot coloring.

        Resolution order:
        1. If color_by == "embeddings", use first column of the embedding itself.
        2. If color_by is set to another key, look in scores[key], then embeddings[key].
        3. Use embeddings['label'] if available.
        4. Fall back to dataset.get_labels().

        Args:
            dataset: Dataset object with optional get_labels() method.
            embeddings: Dictionary potentially containing 'label' or 'scores'.

        Returns:
            Numpy array of colors/labels, or None if unavailable.
        """
        color_array = None

        if self.color_by == "embeddings":
            emb = embeddings.get("embeddings")
            if emb is not None:
                if hasattr(emb, "detach"):
                    emb = emb.detach()
                if hasattr(emb, "cpu"):
                    emb = emb.cpu()
                if hasattr(emb, "numpy"):
                    emb = emb.numpy()
                emb = np.asarray(emb)
                color_array = emb[:, 0] if emb.ndim == 2 else emb
                logger.info("Using embedding values (first component) for coloring.")
            else:
                logger.warning(
                    "color_by='embeddings' but no embeddings found; "
                    "falling back to label-based coloring."
                )

        elif self.color_by is not None:
            # Try to get from scores dict first
            scores = embeddings.get("scores")
            if scores is not None and isinstance(scores, dict):
                color_array = scores.get(self.color_by)

            # Fall back to direct key
            if color_array is None:
                color_array = embeddings.get(self.color_by)

            if color_array is None:
                logger.warning(
                    f"Coloring key '{self.color_by}' not found; "
                    "falling back to label-based coloring."
                )
            else:
                logger.info(f"Using '{self.color_by}' for coloring.")

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

    @staticmethod
    def _category_color_basis(color_array):
        """Compute (unique_labels, numeric, vmin, vmax) for categorical coloring.

        ``vmin``/``vmax`` mirror the implicit ``Normalize`` the categorical
        scatter applies: ``ax.scatter(c=color_array, cmap=...)`` with no explicit
        norm normalizes over ``[min, max]`` of the values.
        """
        unique_labels = sorted(set(color_array))
        arr = np.asarray(color_array)
        numeric = arr.size > 0 and np.issubdtype(arr.dtype, np.number)
        vmin = float(arr.min()) if numeric else 0.0
        vmax = float(arr.max()) if numeric else 0.0
        return unique_labels, numeric, vmin, vmax

    def _category_color(self, lbl, cmap_info, unique_labels, numeric, vmin, vmax):
        """Resolve one categorical label to the exact color the scatter draws.

        Shared by the legend (``_add_categorical_legend_from_data``) and the
        exported sidecar (``_resolve_category_colors``) so the two cannot drift
        (issue #277). For non-dict colormaps the categorical scatter lets
        matplotlib's ``Normalize`` map each point by **value** over
        ``[vmin, vmax]``; indexing by **rank** instead disagrees for
        non-uniformly-spaced integer labels (e.g. ``[0, 1, 10]``: value
        ``1 -> 0.1`` but rank ``1 -> 0.5``). We mirror the value normalization so
        the PNG, its legend, and the sidecar agree. The dict-cmap path is exact
        and is used as the reference.
        """
        if isinstance(cmap_info.cmap, dict):
            return cmap_info.cmap.get(lbl, "#808080")
        cmap = (
            cmap_info.cmap
            if isinstance(cmap_info.cmap, ListedColormap)
            else plt.colormaps.get_cmap(cmap_info.cmap)
        )
        if numeric:
            # Match the scatter's value normalization over [vmin, vmax].
            frac = (float(lbl) - vmin) / ((vmax - vmin) or 1.0)
        else:
            # Non-numeric labels never reach the value-normalized scatter path;
            # fall back to rank so legend/sidecar stay defined and consistent.
            frac = unique_labels.index(lbl) / max(1, len(unique_labels) - 1)
        return cmap(frac)

    def _add_categorical_legend_from_data(
        self,
        ax: plt.Axes,
        labels: np.ndarray,
        cmap_info: ColormapInfo,
    ) -> None:
        """
        Add categorical legend when no label_names provided, using unique labels from data.

        Colors are resolved via :meth:`_category_color` so each legend swatch
        matches the color the scatter actually draws for that label (issue #277).

        Args:
            ax: Matplotlib axes object.
            labels: Array of label values.
            cmap_info: ColormapInfo containing the colormap.
        """
        unique_labels, numeric, vmin, vmax = self._category_color_basis(labels)

        patches = [
            Patch(
                facecolor=self._category_color(
                    lbl, cmap_info, unique_labels, numeric, vmin, vmax
                ),
                label=str(lbl),
            )
            for lbl in unique_labels
        ]

        ax.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(8, len(patches)),
            fontsize=8,
        )

    def _resolve_category_colors(self, color_array, cmap_info) -> Dict[str, str]:
        """Map each unique categorical label to the hex color the scatter draws.

        Uses the same value-normalized resolver as the legend
        (:meth:`_category_color`) so the PNG, its legend, and the exported
        sidecar agree even for non-uniformly-spaced labels (issue #277).
        """
        unique_labels, numeric, vmin, vmax = self._category_color_basis(color_array)
        return {
            str(lbl): mcolors.to_hex(
                self._category_color(lbl, cmap_info, unique_labels, numeric, vmin, vmax)
            )
            for lbl in unique_labels
        }

    def _export_plot_data(
        self, embeddings_to_plot, color_array, cmap_info, base_name
    ) -> None:
        """Write ``{base_name}.npz`` (coords + color_values) and ``{base_name}.json`` (metadata)."""
        npz_path = os.path.join(self.save_dir, f"{base_name}.npz")
        json_path = os.path.join(self.save_dir, f"{base_name}.json")
        os.makedirs(self.save_dir, exist_ok=True)
        arrays = {"coords": np.asarray(embeddings_to_plot, dtype=np.float32)}
        meta = {
            "color_by": self.color_by or self.label_col,
            "is_categorical": bool(cmap_info.is_categorical),
            "n_points": int(embeddings_to_plot.shape[0]),
        }
        if color_array is not None:
            arrays["color_values"] = np.asarray(color_array)
            if cmap_info.is_categorical:
                meta["category_colors"] = self._resolve_category_colors(
                    color_array, cmap_info
                )
            else:
                finite = np.asarray(color_array, dtype=float)
                finite = finite[np.isfinite(finite)]
                if len(finite):
                    meta["vmin"] = float(finite.min())
                    meta["vmax"] = float(finite.max())
                # Derive the real colormap name for non-string cmaps (Colormap/
                # ListedColormap) so the sidecar records what the PNG actually used,
                # instead of silently falling back to "viridis".
                cm = cmap_info.cmap
                meta["cmap"] = cm if isinstance(cm, str) else getattr(cm, "name", "viridis")
        np.savez_compressed(npz_path, **arrays)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved plot data sidecar to {npz_path} and {json_path}")
        self.register_output("embedding_data_path", npz_path)
        self.register_output("embedding_data_meta_path", json_path)

    def _upload_to_wandb(self) -> None:
        """Upload plot to WandB if enabled and run is active."""
        if self.enable_wandb_upload and wandb is not None and wandb.run is not None:
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
        embeddings: Optional[dict] = None,
    ) -> str:
        """
        Create and save the embedding scatter plot using matplotlib.

        Args:
            dataset: Dataset for colormap information.
            embeddings_to_plot: 2D array of shape (n_samples, 2).
            color_array: Array of colors/labels for each point.
            embeddings: Full embeddings dict for metric viz metadata lookup.

        Returns:
            Path to the saved plot file.
        """
        cmap_info = self._get_colormap(dataset, embeddings)

        # Auto-detect categorical/continuous when coloring by embeddings
        if (
            self.color_by == "embeddings"
            and self.is_categorical_override is None
            and color_array is not None
        ):
            detected = self._is_categorical(color_array)
            cmap_info = ColormapInfo(
                cmap=cmap_info.cmap,
                label_names=cmap_info.label_names,
                label_format=cmap_info.label_format,
                is_categorical=detected,
            )
            logger.debug(
                f"Auto-detected color_by='embeddings' as "
                f"{'categorical' if detected else 'continuous'}"
            )

        # Resolve semantic hints to concrete colormap/labels using actual data
        cmap, label_names, is_categorical = resolve_colormap(cmap_info, color_array)

        # Update cmap_info with resolved values for legend generation
        cmap_info = ColormapInfo(
            cmap=cmap,
            label_names=label_names,
            label_format=cmap_info.label_format,
            is_categorical=is_categorical,
        )

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
            if not cmap_info.is_categorical and self.color_by:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(f"{self.color_by} Score")
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

        # Machine-readable sidecar (paired with the PNG via the same timestamp)
        if self.export_plot_data:
            self._export_plot_data(
                embeddings_to_plot,
                color_array,
                cmap_info,
                f"embedding_data_{self.experiment_name}_{timestamp}",
            )

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
        path = self._plot_embeddings(dataset, emb2d, colors, embeddings)
        self.register_output("embedding_plot_path", path)
        return self.callback_outputs
