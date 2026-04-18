"""Synthetic interventions on data arrays.

Pure-function transforms that operate on ``(n, d)`` arrays. Use upstream of
the ``run()`` API by passing ``input_data=transformed_array``.

Currently provides:
    - ``density_spike``: inject a localized density multiplier into a
      manifold for spike-and-recover diagnostic experiments.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def density_spike(
    data: np.ndarray,
    center: Optional[np.ndarray] = None,
    center_idx: Optional[int] = None,
    radius: float = 0.5,
    multiplier: int = 5,
    noise: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a localized density spike into a point cloud.

    Points within ``radius`` of ``center`` are duplicated ``multiplier - 1``
    times (so the local density is multiplied by ``multiplier``) and each
    duplicate is jittered by isotropic Gaussian noise with standard
    deviation ``noise``.

    Args:
        data: ``(n, d)`` original point cloud.
        center: ``(d,)`` ambient-space coordinates of the spike center.
            Mutually exclusive with ``center_idx``.
        center_idx: Index into ``data`` to use as the spike center.
            Mutually exclusive with ``center``.
        radius: Euclidean radius defining the spike region.
        multiplier: Final density multiplier in the spike region. Must be
            >= 1; ``multiplier == 1`` is a no-op (no duplicates added).
        noise: Standard deviation of isotropic Gaussian jitter applied to
            each duplicated point.
        random_state: Seed for the jitter RNG.

    Returns:
        ``(augmented_data, spike_label)`` where ``augmented_data`` is the
        concatenation of the original points and the duplicates, and
        ``spike_label`` is an ``int8`` array of length ``n + n_duplicates``
        with ``0`` for original points and ``1`` for spike duplicates.

    Raises:
        ValueError: If neither or both of ``center`` / ``center_idx`` are
            provided, or if ``multiplier < 1``.
    """
    if (center is None) == (center_idx is None):
        raise ValueError("Provide exactly one of `center` or `center_idx`.")
    if multiplier < 1:
        raise ValueError(f"multiplier must be >= 1, got {multiplier}")

    data = np.asarray(data)
    n, d = data.shape

    if center is None:
        center = data[center_idx]
    center = np.asarray(center, dtype=data.dtype)
    if center.shape != (d,):
        raise ValueError(f"center has shape {center.shape}, expected ({d},)")

    radii = np.linalg.norm(data - center, axis=1)
    region_mask = radii <= radius
    n_region = int(region_mask.sum())

    if n_region == 0:
        logger.warning(
            f"density_spike: no points within radius {radius} of center; returning data unchanged"
        )
        return data.copy(), np.zeros(n, dtype=np.int8)

    n_duplicates_per_point = multiplier - 1
    n_duplicates = n_region * n_duplicates_per_point

    if n_duplicates == 0:
        # multiplier == 1 — return originals with all-zero label
        return data.copy(), np.zeros(n, dtype=np.int8)

    rng = np.random.default_rng(random_state)
    region_points = data[region_mask]
    duplicates = np.repeat(region_points, n_duplicates_per_point, axis=0)
    if noise > 0:
        duplicates = duplicates + rng.normal(0.0, noise, size=duplicates.shape).astype(data.dtype)

    augmented = np.concatenate([data, duplicates], axis=0)
    spike_label = np.concatenate(
        [np.zeros(n, dtype=np.int8), np.ones(n_duplicates, dtype=np.int8)]
    )

    logger.info(
        f"density_spike: {n_region} points in region "
        f"(radius={radius}); added {n_duplicates} duplicates "
        f"(multiplier={multiplier}, noise={noise})"
    )
    return augmented, spike_label
