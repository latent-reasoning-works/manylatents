# manylatents/gauge/tests/test_trajectory.py
import pytest
import numpy as np
from manylatents.gauge.trajectory import TrajectoryVisualizer


def make_trajectory(n_steps: int, n_samples: int, seed: int = 42):
    """Create a fake trajectory of diffusion operators."""
    rng = np.random.default_rng(seed)
    trajectory = []
    for step in range(n_steps):
        # Gradually changing operator
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

    assert embedding.shape == (10, 2)  # 10 steps, 2 dims


def test_trajectory_visualizer_distances():
    """Should compute pairwise distances between operators."""
    trajectory = make_trajectory(n_steps=5, n_samples=20)

    viz = TrajectoryVisualizer()
    distances = viz.compute_distances(trajectory)

    assert distances.shape == (5, 5)
    # Diagonal should be zero
    np.testing.assert_allclose(np.diag(distances), 0, atol=1e-10)
    # Should be symmetric
    np.testing.assert_allclose(distances, distances.T)


def test_trajectory_visualizer_spread():
    """Should compute spread metric."""
    trajectory = make_trajectory(n_steps=5, n_samples=20)

    viz = TrajectoryVisualizer()
    spread = viz.compute_spread(trajectory)

    assert isinstance(spread, float)
    assert spread > 0  # Non-trivial trajectory should have positive spread
