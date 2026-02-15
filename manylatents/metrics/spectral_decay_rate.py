"""Spectral Decay Rate metric.

Fits exponential decay lambda_i ~ exp(-rate * i) to the top-k eigenvalues.
Faster decay indicates lower effective dimensionality.
"""
import logging
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.utils.metrics import compute_eigenvalues

logger = logging.getLogger(__name__)


def SpectralDecayRate(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    top_k: int = 20,
    cache: Optional[dict] = None,
) -> float:
    """Fit exponential decay to the eigenvalue spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix().
        top_k: Number of eigenvalues to fit.
        cache: Shared cache dict. Pass through to compute_eigenvalues().

    Returns:
        float: Decay rate (positive = decaying), or nan if unavailable.
    """
    eigenvalues = compute_eigenvalues(module, cache=cache)
    if eigenvalues is None or len(eigenvalues) < 3:
        return float("nan")

    eigenvalues = eigenvalues[:top_k]

    # Only use positive eigenvalues for log fit
    pos_mask = eigenvalues > 0
    if pos_mask.sum() < 3:
        return float("nan")

    eigs_pos = eigenvalues[pos_mask]
    log_eigs = np.log(eigs_pos)
    indices = np.arange(len(eigs_pos))

    # Linear fit: log(lambda_i) = -rate * i + intercept
    coeffs = np.polyfit(indices, log_eigs, 1)
    rate = float(-coeffs[0])  # Negate so positive = decaying

    logger.info(f"SpectralDecayRate: {rate:.4f} (top_k={top_k})")
    return rate
