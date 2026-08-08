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

    def test_momentum_beta_zero_matches_unsmoothed(self):
        """momentum_beta=0.0 (default) reproduces the plain velocity field."""
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

        torch.manual_seed(0)
        func = MIOFlowODEFunc(input_dim=5, hidden_dim=16, momentum_beta=0.0)
        x = torch.randn(10, 5)
        dx1 = func(torch.tensor(0.0), x)
        func.reset_momentum()
        dx2 = func(torch.tensor(0.0), x)
        assert torch.allclose(dx1, dx2)

    def test_momentum_beta_smooths_across_calls(self):
        """A non-zero momentum_beta blends successive velocity predictions."""
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

        torch.manual_seed(0)
        func = MIOFlowODEFunc(input_dim=5, hidden_dim=16, momentum_beta=0.9)
        func.reset_momentum()
        x = torch.randn(10, 5)
        dx_t0 = func(torch.tensor(0.0), x)
        dx_t1 = func(torch.tensor(1.0), x)
        # With heavy momentum, the second call should be pulled toward the first.
        raw_net_out = func.net(
            torch.cat([torch.tensor(1.0).expand(x.size(0), 1), x], dim=-1)
        )
        assert not torch.allclose(dx_t1, raw_net_out)


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

    def test_density_loss_respects_top_k_and_hinge_value(self):
        """Density loss accepts the previously-hardcoded top_k/hinge_value kwargs."""
        from manylatents.algorithms.lightning.networks.mioflow_net import mioflow_density_loss

        source = torch.randn(30, 5)
        target = torch.randn(30, 5)
        loss_default = mioflow_density_loss(source, target)
        loss_custom = mioflow_density_loss(source, target, top_k=3, hinge_value=0.5)
        assert loss_custom.shape == ()
        # A larger hinge value can only shrink (never grow) the clamped penalty.
        assert loss_custom.item() <= loss_default.item() + 1e-6


class TestPreprocessor:
    """Tests for the GAGA Preprocessor normalization buffers."""

    def test_normalize_unnormalize_round_trip(self):
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        torch.manual_seed(0)
        x = torch.randn(20, 5) * 3 + 1
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        pre = Preprocessor(mean=mean, std=std, dist_std=1.0)

        x_norm = pre.normalize(x)
        x_back = pre.unnormalize(x_norm)
        assert torch.allclose(x_back, x, atol=1e-5)

    def test_normalize_dist_scales_by_dist_std(self):
        from manylatents.algorithms.lightning.networks.gaga_net import Preprocessor

        pre = Preprocessor(mean=0.0, std=1.0, dist_std=2.0)
        d = torch.tensor([1.0, 2.0, 4.0])
        assert torch.allclose(pre.normalize_dist(d), d / 2.0)


class TestGAGANetwork:
    """Tests for the GAGA encoder/decoder network."""

    def test_forward_shapes(self):
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net = GAGANetwork(input_dim=20, latent_dim=4, hidden_dims=[16, 8])
        x = torch.randn(10, 20)
        x_hat, z = net(x)
        assert z.shape == (10, 4)
        assert x_hat.shape == (10, 20)

    def test_hidden_dims_accepts_int_or_list(self):
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        net_int = GAGANetwork(input_dim=10, latent_dim=3, hidden_dims=16)
        net_list = GAGANetwork(input_dim=10, latent_dim=3, hidden_dims=[16])
        assert net_int.hidden_dims == net_list.hidden_dims == [16]

    def test_encode_decode_match_forward(self):
        from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork

        torch.manual_seed(0)
        net = GAGANetwork(input_dim=8, latent_dim=2)
        x = torch.randn(5, 8)
        x_hat, z = net(x)
        assert torch.allclose(net.encode(x), z)
        assert torch.allclose(net.decode(z), x_hat)


class TestGAGALosses:
    """Tests for the GAGA distance-preservation and reconstruction losses."""

    def test_distance_loss_zero_when_matching(self):
        from manylatents.algorithms.lightning.networks.gaga_net import gaga_distance_loss

        torch.manual_seed(0)
        z = torch.randn(15, 3)
        gt = torch.nn.functional.pdist(z)
        loss = gaga_distance_loss(z, gt)
        assert loss.item() < 1e-6

    def test_distance_loss_positive_when_mismatched(self):
        from manylatents.algorithms.lightning.networks.gaga_net import gaga_distance_loss

        torch.manual_seed(0)
        z = torch.randn(15, 3)
        gt = torch.nn.functional.pdist(z) + 5.0
        loss = gaga_distance_loss(z, gt)
        assert loss.item() > 1.0

    def test_reconstruction_loss_zero_when_identical(self):
        from manylatents.algorithms.lightning.networks.gaga_net import gaga_reconstruction_loss

        x = torch.randn(10, 5)
        assert gaga_reconstruction_loss(x, x.clone()).item() < 1e-8

    def test_reconstruction_loss_positive_when_different(self):
        from manylatents.algorithms.lightning.networks.gaga_net import gaga_reconstruction_loss

        x = torch.randn(10, 5)
        y = torch.randn(10, 5)
        assert gaga_reconstruction_loss(x, y).item() > 0
