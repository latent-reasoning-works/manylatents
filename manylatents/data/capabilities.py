"""Dataset capability discovery.

Provides runtime inspection of dataset ground truth interfaces.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DatasetCapabilities(Protocol):
    """Protocol for datasets with optional ground truth methods."""

    def get_gt_dists(self) -> Any: ...
    def get_graph(self) -> Any: ...
    def get_labels(self) -> Any: ...


def get_capabilities(dataset: Any) -> dict[str, bool | str]:
    """Inspect a dataset and return which ground truth interfaces it supports.

    Args:
        dataset: Any dataset object.

    Returns:
        Dict with keys: gt_dists, graph, labels, centers (bool),
        and gt_type (str: "manifold"|"graph"|"euclidean"|"unknown").
    """
    caps: dict[str, bool | str] = {
        "gt_dists": hasattr(dataset, "get_gt_dists") and callable(dataset.get_gt_dists),
        "graph": hasattr(dataset, "get_graph") and callable(dataset.get_graph),
        "labels": hasattr(dataset, "get_labels") and callable(dataset.get_labels),
        "centers": hasattr(dataset, "get_centers") and callable(dataset.get_centers),
    }

    if caps["gt_dists"]:
        cls_name = type(dataset).__name__
        if "DLA" in cls_name or "Tree" in cls_name:
            caps["gt_type"] = "graph"
        elif "Blob" in cls_name:
            caps["gt_type"] = "euclidean"
        else:
            caps["gt_type"] = "manifold"
    else:
        caps["gt_type"] = "unknown"

    return caps


def log_capabilities(dataset: Any) -> dict[str, bool | str]:
    """Discover and log dataset capabilities."""
    caps = get_capabilities(dataset)
    logger.info(
        "Dataset capabilities: "
        + ", ".join(f"{k}={v}" for k, v in caps.items())
    )
    return caps
