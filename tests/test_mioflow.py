"""Tests for MIOFlow network components and LightningModule."""
import numpy as np
import pytest
import torch

pytest.importorskip("torchdiffeq")
pot = pytest.importorskip("ot")


class TestMIOFlowODEFunc:
    """Tests for the ODEFunc neural network."""

    def test_forward_shape(self):
        """ODEFunc output matches input spatial dims."""
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

        func = MIOFlowODEFunc(input_dim=20, hidden_dim=64)
        x = torch.randn(50, 20)
        t = torch.tensor(0.5)
        dx = func(t, x)
        assert dx.shape == (50, 20)

    def test_time_dependence(self):
        """ODEFunc output changes with different time values."""
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

        func = MIOFlowODEFunc(input_dim=5, hidden_dim=32)
        x = torch.randn(10, 5)
        dx_t0 = func(torch.tensor(0.0), x)
        dx_t1 = func(torch.tensor(1.0), x)
        assert not torch.allclose(dx_t0, dx_t1), "ODEFunc should be time-dependent"


class TestMIOFlowLosses:
    """Tests for OT, energy, and density losses."""

    def test_ot_loss_returns_scalar(self):
        """OT loss returns a scalar tensor."""
        from manylatents.algorithms.lightning.networks.mioflow_net import mioflow_ot_loss

        source = torch.randn(30, 5)
        target = torch.randn(30, 5)
        loss = mioflow_ot_loss(source, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_ot_loss_zero_for_identical(self):
        """OT loss is zero when source == target."""
        from manylatents.algorithms.lightning.networks.mioflow_net import mioflow_ot_loss

        x = torch.randn(20, 5)
        loss = mioflow_ot_loss(x, x.clone())
        assert loss.item() < 1e-5

    def test_energy_loss_returns_scalar(self):
        """Energy loss returns a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.mioflow_net import (
            MIOFlowODEFunc,
            mioflow_energy_loss,
        )

        func = MIOFlowODEFunc(input_dim=5, hidden_dim=32)
        x0 = torch.randn(10, 5)
        t_seq = torch.linspace(0, 1, 5)
        loss = mioflow_energy_loss(func, x0, t_seq)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_density_loss_returns_scalar(self):
        """Density loss returns a non-negative scalar."""
        from manylatents.algorithms.lightning.networks.mioflow_net import mioflow_density_loss

        source = torch.randn(30, 5)
        target = torch.randn(30, 5)
        loss = mioflow_density_loss(source, target)
        assert loss.shape == ()
        assert loss.item() >= 0
