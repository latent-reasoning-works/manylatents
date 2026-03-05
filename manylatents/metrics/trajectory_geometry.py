"""Trajectory geometry primitives (Zhou et al.).

Provides velocity, cosine velocity, and Menger curvature for ordered
embedding sequences.  The undecorated helpers operate on (T, D) arrays;
the registered wrappers handle trace-aware grouping.
"""

from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


# ---------------------------------------------------------------------------
# Pure-numpy helpers
# ---------------------------------------------------------------------------


def compute_velocity(embeddings: np.ndarray) -> np.ndarray:
    """Finite differences: Δy_t = y_t - y_{t-1}.

    Args:
        embeddings: Array of shape (T, D).

    Returns:
        Array of shape (T-1, D).
    """
    return embeddings[1:] - embeddings[:-1]


def compute_cosine_velocity(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive embeddings.

    Args:
        embeddings: Array of shape (T, D).

    Returns:
        Array of shape (T-1,) with values in [0, 2].
    """
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    normed = embeddings / norms
    cos_sim = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1.0 - cos_sim


def compute_menger_curvature(embeddings: np.ndarray) -> np.ndarray:
    """Menger curvature via Gram determinant (Zhou et al. Prop. C.8).

    For each triple (y_{t-1}, y_t, y_{t+1}):
        u = y_t - y_{t-1},  v = y_{t+1} - y_t
        κ_t = 2 * sqrt(1 - cos²(u, v)) / ‖y_{t+1} - y_{t-1}‖

    Args:
        embeddings: Array of shape (T, D) with T >= 3.

    Returns:
        Array of shape (T-2,) of non-negative curvature values.
    """
    u = embeddings[1:-1] - embeddings[:-2]   # (T-2, D)
    v = embeddings[2:] - embeddings[1:-1]     # (T-2, D)

    u_norm = np.maximum(np.linalg.norm(u, axis=1), 1e-8)
    v_norm = np.maximum(np.linalg.norm(v, axis=1), 1e-8)

    cos_uv = np.sum(u * v, axis=1) / (u_norm * v_norm)
    cos_uv = np.clip(cos_uv, -1.0, 1.0)

    sin_uv = np.sqrt(np.maximum(1.0 - cos_uv ** 2, 0.0))

    chord = np.maximum(np.linalg.norm(embeddings[2:] - embeddings[:-2], axis=1), 1e-8)

    return 2.0 * sin_uv / chord


# ---------------------------------------------------------------------------
# Registered metric wrappers (trace-aware)
# ---------------------------------------------------------------------------


def _get_trace_ids(dataset) -> Optional[np.ndarray]:
    """Extract per-step trace IDs from a dataset, if available."""
    if dataset is None:
        return None
    ids = getattr(dataset, "step_trace_ids", None)
    if ids is None and hasattr(dataset, "get_labels"):
        ids = dataset.get_labels()
    return ids


def _per_trace_mean(embeddings, dataset, metric_fn):
    """Compute metric_fn per trace, return overall mean."""
    trace_ids = _get_trace_ids(dataset)
    if trace_ids is None:
        vals = metric_fn(embeddings)
        return float(np.mean(vals)) if vals.size > 0 else 0.0

    unique_ids = np.unique(trace_ids)
    means = []
    for tid in unique_ids:
        mask = trace_ids == tid
        trace_emb = embeddings[mask]
        if len(trace_emb) < 2:
            continue
        vals = metric_fn(trace_emb)
        if vals.size > 0:
            means.append(float(np.mean(vals)))
    return float(np.mean(means)) if means else 0.0


@register_metric(
    aliases=["trajectory_velocity"],
    default_params={},
    description="Mean cosine velocity across all traces",
)
def TrajectoryVelocity(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
) -> float:
    """Mean cosine velocity across all traces."""
    return _per_trace_mean(embeddings, dataset, compute_cosine_velocity)


@register_metric(
    aliases=["trajectory_curvature", "menger_curvature"],
    default_params={},
    description="Mean Menger curvature across all traces",
)
def TrajectoryCurvature(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
) -> float:
    """Mean Menger curvature across all traces."""
    def _curvature(emb):
        if len(emb) < 3:
            return np.array([])
        return compute_menger_curvature(emb)
    return _per_trace_mean(embeddings, dataset, _curvature)
