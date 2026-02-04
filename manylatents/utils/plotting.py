"""Plotting utilities for dynamic colormap and label generation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from manylatents.callbacks.embedding.base import ColormapInfo

logger = logging.getLogger(__name__)

# Default categorical color palette (colorblind-friendly, up to 10 colors)
DEFAULT_CATEGORICAL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def generate_categorical_colors(n_categories: int) -> List[str]:
    """
    Generate colors for n categories.

    Uses default palette for up to 10 categories, falls back to
    matplotlib colormap for more.

    Args:
        n_categories: Number of distinct categories.

    Returns:
        List of hex color strings.
    """
    if n_categories <= len(DEFAULT_CATEGORICAL_COLORS):
        return DEFAULT_CATEGORICAL_COLORS[:n_categories]

    # Fall back to tab20 for more categories
    cmap = plt.colormaps.get_cmap("tab20")
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
        )
        for c in [cmap(i / n_categories) for i in range(n_categories)]
    ]


def resolve_colormap(
    cmap_info: ColormapInfo,
    color_array: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[Union[int, str], str], bool]:
    """
    Resolve a ColormapInfo into concrete colormap, label names, and colors.

    This function takes semantic colormap hints and actual data to produce
    concrete visualization parameters. It handles:
    - Dynamic color generation for categorical data
    - Dynamic label generation from label_format
    - Passthrough for already-concrete colormaps

    Args:
        cmap_info: Semantic colormap information from metric or dataset.
        color_array: Optional array of values to be colored. Used to determine
                    unique categories for dynamic generation.

    Returns:
        Tuple of (cmap, label_names, is_categorical) where:
        - cmap: Concrete colormap (str, dict, or ListedColormap)
        - label_names: Dict mapping values to display names
        - is_categorical: Whether to render as categorical
    """
    cmap = cmap_info.cmap
    label_names = cmap_info.label_names
    is_categorical = cmap_info.is_categorical

    # If we have color_array and categorical data, we may need to generate labels/colors
    if color_array is not None and is_categorical:
        unique_vals = sorted(set(color_array))
        n_unique = len(unique_vals)

        # Generate label names if not provided but format is
        if label_names is None and cmap_info.label_format is not None:
            label_names = {v: cmap_info.label_format.format(v) for v in unique_vals}
            logger.debug(f"Generated label names from format: {label_names}")

        # Generate colors if cmap is "categorical" hint
        if cmap == "categorical":
            colors = generate_categorical_colors(n_unique)
            cmap = {v: colors[i] for i, v in enumerate(unique_vals)}
            logger.debug(f"Generated categorical colormap for {n_unique} categories")

    return cmap, label_names, is_categorical


def apply_colormap_to_array(
    color_array: np.ndarray,
    cmap: Union[str, Dict[Union[int, str], str], ListedColormap],
    is_categorical: bool = True,
) -> Union[List[str], np.ndarray]:
    """
    Apply a colormap to an array of values.

    Args:
        color_array: Array of values to map to colors.
        cmap: Colormap to use (dict, string name, or ListedColormap).
        is_categorical: Whether data is categorical.

    Returns:
        List of color strings (categorical dict cmap) or the original array
        (for matplotlib to handle with string/ListedColormap).
    """
    if isinstance(cmap, dict):
        default_color = "#808080"  # Gray for unknown
        return [cmap.get(v, default_color) for v in color_array]

    # For string cmaps and ListedColormap, return array for matplotlib to handle
    return color_array


def merge_colormap_info(
    base: Optional[ColormapInfo],
    override: Optional[ColormapInfo] = None,
    cmap_override: Optional[str] = None,
    is_categorical_override: Optional[bool] = None,
    label_names_override: Optional[Dict] = None,
    label_format_override: Optional[str] = None,
) -> ColormapInfo:
    """
    Merge colormap info with optional overrides.

    Resolution order (highest to lowest priority):
    1. Individual field overrides (cmap_override, etc.)
    2. Full ColormapInfo override
    3. Base ColormapInfo
    4. Defaults

    Args:
        base: Base colormap info (e.g., from metric).
        override: Full override (e.g., from user config).
        cmap_override: Override just the cmap.
        is_categorical_override: Override just is_categorical.
        label_names_override: Override just label_names.
        label_format_override: Override just label_format.

    Returns:
        Merged ColormapInfo.
    """
    # Start with defaults
    result = ColormapInfo(cmap="viridis", is_categorical=True)

    # Apply base
    if base is not None:
        result = ColormapInfo(
            cmap=base.cmap,
            label_names=base.label_names,
            label_format=base.label_format,
            is_categorical=base.is_categorical,
        )

    # Apply full override
    if override is not None:
        result = ColormapInfo(
            cmap=override.cmap if override.cmap is not None else result.cmap,
            label_names=override.label_names if override.label_names is not None else result.label_names,
            label_format=override.label_format if override.label_format is not None else result.label_format,
            is_categorical=override.is_categorical,
        )

    # Apply individual overrides
    if cmap_override is not None:
        result = ColormapInfo(
            cmap=cmap_override,
            label_names=result.label_names,
            label_format=result.label_format,
            is_categorical=result.is_categorical,
        )
    if is_categorical_override is not None:
        result = ColormapInfo(
            cmap=result.cmap,
            label_names=result.label_names,
            label_format=result.label_format,
            is_categorical=is_categorical_override,
        )
    if label_names_override is not None:
        result = ColormapInfo(
            cmap=result.cmap,
            label_names=label_names_override,
            label_format=result.label_format,
            is_categorical=result.is_categorical,
        )
    if label_format_override is not None:
        result = ColormapInfo(
            cmap=result.cmap,
            label_names=result.label_names,
            label_format=label_format_override,
            is_categorical=result.is_categorical,
        )

    return result
