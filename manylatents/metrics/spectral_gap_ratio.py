"""Spectral Gap Ratio metric.

Computes lambda_1 / lambda_2 from the affinity matrix eigenvalue spectrum.
A large gap indicates clear separation between the dominant mode and the rest.
"""
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SpectralGapRatio(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    _eigenvalue_cache: Optional[Dict[Tuple, np.ndarray]] = None,
) -> float:
    """Compute ratio of first to second eigenvalue of the affinity spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused, kept for protocol).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix() method.
        _eigenvalue_cache: Shared eigenvalue cache from evaluate().

    Returns:
        float: lambda_1 / lambda_2, or nan if unavailable.
    """
    eigenvalues = _get_eigenvalues(module, _eigenvalue_cache)
    if eigenvalues is None or len(eigenvalues) < 2:
        return float("nan")

    if eigenvalues[1] == 0:
        return float("inf")

    ratio = float(eigenvalues[0] / eigenvalues[1])
    logger.info(f"SpectralGapRatio: {ratio:.4f}")
    return ratio


def _get_eigenvalues(
    module: Optional[LatentModule],
    cache: Optional[Dict[Tuple, np.ndarray]],
) -> Optional[np.ndarray]:
    """Get eigenvalues from cache or compute from module."""
    if cache is not None:
        # Prefer full spectrum
        for key in [(True, None), (True, 25)]:
            if key in cache:
                return cache[key]
        # Take any cached entry
        if cache:
            return next(iter(cache.values()))

    if module is not None:
        try:
            A = module.affinity_matrix(use_symmetric=True)
            eigs = np.linalg.eigvalsh(A)
            return np.sort(eigs)[::-1]
        except (NotImplementedError, AttributeError):
            warnings.warn(
                f"SpectralGapRatio: {type(module).__name__} does not expose affinity_matrix.",
                RuntimeWarning,
            )

    return None
