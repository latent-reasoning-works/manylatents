"""Spectral Gap Ratio metric.

Computes lambda_1 / lambda_2 from the affinity matrix eigenvalue spectrum.
A large gap indicates clear separation between the dominant mode and the rest.
"""
import logging
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_eigenvalues

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["spectral_gap_ratio", "spectral_gap"],
    default_params={},
    description="Ratio of first to second eigenvalue of the diffusion operator",
)
def SpectralGapRatio(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
) -> float:
    """Compute ratio of first to second eigenvalue of the affinity spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused, kept for protocol).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix() method.
        cache: Shared cache dict. Pass through to compute_eigenvalues().

    Returns:
        float: lambda_1 / lambda_2, or nan if unavailable.
    """
    eigenvalues = compute_eigenvalues(module, cache=cache)
    if eigenvalues is None or len(eigenvalues) < 2:
        return float("nan")

    if eigenvalues[1] == 0:
        return float("inf")

    ratio = float(eigenvalues[0] / eigenvalues[1])
    logger.info(f"SpectralGapRatio: {ratio:.4f}")
    return ratio
