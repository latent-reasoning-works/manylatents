"""Dataset Topology Descriptor metric.

Aggregates spectral and topological properties of a dataset+module combination
into a descriptive dictionary. Uses eigenvalue cache and dataset capabilities.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.data.capabilities import get_capabilities
from manylatents.utils.metrics import compute_eigenvalues

logger = logging.getLogger(__name__)


def DatasetTopologyDescriptor(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
) -> dict:
    """Compute a descriptor of the dataset's topological properties.

    Args:
        embeddings: Low-dimensional embeddings (used for dimensionality).
        dataset: Dataset object for capabilities inspection.
        module: Fitted LatentModule with affinity_matrix().
        cache: Shared cache dict. Pass through to compute_eigenvalues().

    Returns:
        dict with keys: spectral_gap, effective_dim, gt_type, n_samples, n_features.
    """
    result: Dict[str, Any] = {
        "n_samples": embeddings.shape[0],
        "n_features": embeddings.shape[1],
    }

    # Dataset capabilities
    if dataset is not None:
        caps = get_capabilities(dataset)
        result["gt_type"] = caps.get("gt_type", "unknown")
        result["has_labels"] = caps.get("labels", False)
        result["has_gt_dists"] = caps.get("gt_dists", False)
    else:
        result["gt_type"] = "unknown"
        result["has_labels"] = False
        result["has_gt_dists"] = False

    # Spectral properties from eigenvalue cache
    eigenvalues = compute_eigenvalues(module, cache=cache)
    if eigenvalues is not None and len(eigenvalues) >= 2:
        result["spectral_gap"] = float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] != 0 else float("inf")

        # Effective dimensionality: number of eigenvalues > 1% of max
        threshold = 0.01 * eigenvalues[0]
        result["effective_dim"] = int(np.sum(eigenvalues > threshold))

        # Participation ratio of eigenvalues
        eig_sum = np.sum(eigenvalues)
        eig_sq_sum = np.sum(eigenvalues ** 2)
        if eig_sq_sum > 0:
            result["spectral_participation_ratio"] = float(eig_sum ** 2 / eig_sq_sum)
    else:
        result["spectral_gap"] = float("nan")
        result["effective_dim"] = -1
        result["spectral_participation_ratio"] = float("nan")

    logger.info(f"DatasetTopologyDescriptor: {result}")
    return result
