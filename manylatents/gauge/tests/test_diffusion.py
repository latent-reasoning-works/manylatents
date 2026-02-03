# manylatents/gauge/tests/test_diffusion.py
import pytest
import numpy as np
import torch
from manylatents.gauge.diffusion import DiffusionGauge


def test_diffusion_gauge_basic():
    """Gauge should produce a row-stochastic matrix."""
    gauge = DiffusionGauge()
    activations = torch.randn(100, 64)

    diff_op = gauge(activations)

    assert diff_op.shape == (100, 100)
    # Row-stochastic: rows sum to 1
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

    # Still symmetric
    np.testing.assert_allclose(diff_op, diff_op.T, rtol=1e-5)
