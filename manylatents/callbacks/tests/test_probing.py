# manylatents/callbacks/tests/test_probing.py
"""Tests for probing utilities."""
from typing import List, Tuple
import pytest
import numpy as np
import torch
from manylatents.callbacks.probing import (
    probe,
    DiffusionGauge,
    TrajectoryVisualizer,
    compute_multi_model_spread,
)


# =============================================================================
# DiffusionGauge tests
# =============================================================================

def test_diffusion_gauge_basic():
    """Gauge should produce a row-stochastic matrix."""
    gauge = DiffusionGauge()
    activations = torch.randn(100, 64)

    diff_op = gauge(activations)

    assert diff_op.shape == (100, 100)
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_gauge_symmetric():
    """Symmetric mode should produce symmetric matrix."""
    gauge = DiffusionGauge(symmetric=True)
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)
    np.testing.assert_allclose(diff_op, diff_op.T, rtol=1e-5)


def test_diffusion_gauge_deterministic():
    """Same input should produce same output."""
    gauge = DiffusionGauge()
    activations = torch.randn(30, 16)

    diff_op1 = gauge(activations)
    diff_op2 = gauge(activations)

    np.testing.assert_allclose(diff_op1, diff_op2)


def test_diffusion_gauge_cosine_metric():
    """Should work with cosine distance."""
    gauge = DiffusionGauge(metric="cosine")
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_gauge_no_knn():
    """Global bandwidth when knn=None."""
    gauge = DiffusionGauge(knn=None)
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)


def test_diffusion_gauge_alpha_zero():
    """alpha=0 gives graph Laplacian normalization."""
    gauge = DiffusionGauge(alpha=0.0, symmetric=True)
    activations = torch.randn(30, 16)

    diff_op = gauge(activations)

    np.testing.assert_allclose(diff_op, diff_op.T, rtol=1e-5)


# =============================================================================
# probe() dispatch tests
# =============================================================================

def test_probe_dispatch_tensor():
    """probe() should work with torch.Tensor."""
    activations = torch.randn(50, 32)
    diff_op = probe(activations)

    assert diff_op.shape == (50, 50)
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_probe_dispatch_ndarray():
    """probe() should work with np.ndarray."""
    embeddings = np.random.randn(50, 32)
    diff_op = probe(embeddings)

    assert diff_op.shape == (50, 50)
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


# =============================================================================
# Trajectory analysis tests
# =============================================================================

def make_trajectory(n_steps: int, n_samples: int, seed: int = 42):
    """Create a fake trajectory of diffusion operators."""
    rng = np.random.default_rng(seed)
    trajectory = []
    for step in range(n_steps):
        K = rng.random((n_samples, n_samples)) + step * 0.1
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        row_sums = K.sum(axis=1, keepdims=True)
        P = K / row_sums
        trajectory.append((step * 100, P))
    return trajectory


def test_trajectory_visualizer_embed():
    """Should embed trajectory points."""
    trajectory = make_trajectory(n_steps=10, n_samples=30)

    viz = TrajectoryVisualizer(n_components=2)
    embedding = viz.fit_transform(trajectory)

    assert embedding.shape == (10, 2)


def test_trajectory_visualizer_distances():
    """Should compute pairwise distances between operators."""
    trajectory = make_trajectory(n_steps=5, n_samples=20)

    viz = TrajectoryVisualizer()
    distances = viz.compute_distances(trajectory)

    assert distances.shape == (5, 5)
    np.testing.assert_allclose(np.diag(distances), 0, atol=1e-10)
    np.testing.assert_allclose(distances, distances.T)


def test_trajectory_visualizer_spread():
    """Should compute spread metric."""
    trajectory = make_trajectory(n_steps=5, n_samples=20)

    viz = TrajectoryVisualizer()
    spread = viz.compute_spread(trajectory)

    assert isinstance(spread, float)
    assert spread > 0


def make_converging_trajectories(
    n_models: int,
    n_steps: int,
    n_samples: int,
    seed: int = 42,
) -> List[List[Tuple[int, np.ndarray]]]:
    """Create trajectories that converge over time."""
    rng = np.random.default_rng(seed)

    K_target = rng.random((n_samples, n_samples))
    K_target = (K_target + K_target.T) / 2
    np.fill_diagonal(K_target, 0)
    P_target = K_target / K_target.sum(axis=1, keepdims=True)

    trajectories = []
    for model_idx in range(n_models):
        trajectory = []
        K_init = rng.random((n_samples, n_samples))
        K_init = (K_init + K_init.T) / 2
        np.fill_diagonal(K_init, 0)
        P_init = K_init / K_init.sum(axis=1, keepdims=True)

        for step in range(n_steps):
            alpha = step / (n_steps - 1) if n_steps > 1 else 1.0
            P = (1 - alpha) * P_init + alpha * P_target
            trajectory.append((step * 100, P))

        trajectories.append(trajectory)

    return trajectories


def test_trajectory_spread_convergence():
    """Spread should decrease for converging trajectories."""
    trajectories = make_converging_trajectories(
        n_models=3, n_steps=10, n_samples=20
    )

    spreads = compute_multi_model_spread(trajectories)

    assert spreads[-1] < spreads[0]
