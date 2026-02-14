"""Spectral Decay Rate metric.

Fits exponential decay lambda_i ~ exp(-rate * i) to the top-k eigenvalues.
Faster decay indicates lower effective dimensionality.
"""
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SpectralDecayRate(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    top_k: int = 20,
    _eigenvalue_cache: Optional[Dict[Tuple, np.ndarray]] = None,
) -> float:
    """Fit exponential decay to the eigenvalue spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix().
        top_k: Number of eigenvalues to fit.
        _eigenvalue_cache: Shared eigenvalue cache.

    Returns:
        float: Decay rate (positive = decaying), or nan if unavailable.
    """
    eigenvalues = _get_top_eigenvalues(module, _eigenvalue_cache, top_k)
    if eigenvalues is None or len(eigenvalues) < 3:
        return float("nan")

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


def _get_top_eigenvalues(
    module: Optional[LatentModule],
    cache: Optional[Dict[Tuple, np.ndarray]],
    top_k: int,
) -> Optional[np.ndarray]:
    """Get top-k eigenvalues from cache or compute from module."""
    if cache is not None:
        for key in [(True, top_k), (True, None)]:
            if key in cache:
                return cache[key][:top_k]
        if cache:
            eigs = next(iter(cache.values()))
            return eigs[:top_k]

    if module is not None:
        try:
            A = module.affinity_matrix(use_symmetric=True)
            eigs = np.linalg.eigvalsh(A)
            return np.sort(eigs)[::-1][:top_k]
        except (NotImplementedError, AttributeError):
            pass

    return None
