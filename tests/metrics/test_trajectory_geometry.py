"""Tests for trajectory geometry primitives."""

import numpy as np
import pytest

from manylatents.metrics.trajectory_geometry import (
    compute_cosine_velocity,
    compute_menger_curvature,
    compute_velocity,
)


# ---------------------------------------------------------------------------
# compute_velocity
# ---------------------------------------------------------------------------


def test_compute_velocity_shape():
    emb = np.random.randn(10, 32)
    vel = compute_velocity(emb)
    assert vel.shape == (9, 32)


def test_compute_velocity_values():
    emb = np.array([[1.0, 0.0], [3.0, 1.0], [6.0, 5.0]])
    vel = compute_velocity(emb)
    np.testing.assert_array_equal(vel[0], [2.0, 1.0])
    np.testing.assert_array_equal(vel[1], [3.0, 4.0])


# ---------------------------------------------------------------------------
# compute_cosine_velocity
# ---------------------------------------------------------------------------


def test_compute_cosine_velocity_identical():
    v = np.array([1.0, 2.0, 3.0])
    emb = np.tile(v, (5, 1))  # 5 identical rows
    cos_vel = compute_cosine_velocity(emb)
    assert cos_vel.shape == (4,)
    np.testing.assert_allclose(cos_vel, 0.0, atol=1e-7)


def test_compute_cosine_velocity_orthogonal():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    cos_vel = compute_cosine_velocity(emb)
    np.testing.assert_allclose(cos_vel, 1.0, atol=1e-7)


# ---------------------------------------------------------------------------
# compute_menger_curvature
# ---------------------------------------------------------------------------


def test_compute_menger_curvature_straight_line():
    emb = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    curv = compute_menger_curvature(emb)
    assert curv.shape == (2,)
    np.testing.assert_allclose(curv, 0.0, atol=1e-7)


def test_compute_menger_curvature_right_angle():
    emb = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    curv = compute_menger_curvature(emb)
    assert curv.shape == (1,)
    assert curv[0] > 0.0


def test_compute_menger_curvature_shape():
    emb = np.random.randn(10, 16)
    curv = compute_menger_curvature(emb)
    assert curv.shape == (8,)


# ---------------------------------------------------------------------------
# Registered metric wrappers
# ---------------------------------------------------------------------------


def test_registered_metric_callable():
    from manylatents.metrics import compute_metric

    emb = np.random.randn(20, 8)
    result = compute_metric("trajectory_velocity", emb)
    assert isinstance(result, float)
    assert result >= 0.0


def test_registered_curvature_callable():
    from manylatents.metrics import compute_metric

    emb = np.random.randn(20, 8)
    result = compute_metric("trajectory_curvature", emb)
    assert isinstance(result, float)
    assert result >= 0.0
