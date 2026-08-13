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


class TestMIOFlowExtraOutputs:
    """`extra_outputs()` — the only way a fitted MIOFlow's trajectories reach a caller.

    `_generate_trajectories()` runs in `on_train_end()` and stores the paths on the model, but
    `run_experiment()`'s return contract is fixed to embeddings/label/metadata/scores and no
    caller-facing path reaches the fitted instance. `experiment.run_experiment()` already merges
    whatever `extra_outputs()` returns (step 4g), so this method is the whole fix — see #295.
    """

    def _model(self, dim=4):
        from manylatents.algorithms.lightning.mioflow import MIOFlow
        from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

        return MIOFlow(network=MIOFlowODEFunc(input_dim=dim, hidden_dim=16), optimizer=None,
                       n_bins=5, n_trajectories=7)

    def test_empty_before_a_fit_rather_than_raising(self):
        """The engine calls this unconditionally, so an untrained model is an ordinary state.
        Mirrors `Cflows.extra_outputs()`'s guard, asserted the same way its test does."""
        assert self._model().extra_outputs() == {}

    def test_returns_the_trajectories_once_generated(self):
        """The key name matters: `run_experiment()` merges this dict into its results verbatim,
        and `LatentModule`'s default implementation calls the same thing `trajectories`."""
        model = self._model()
        model._trajectories = torch.zeros(5, 7, 4)

        out = model.extra_outputs()

        assert set(out) == {"trajectories"}
        assert out["trajectories"].shape == (5, 7, 4)

    def test_detached_and_on_the_cpu_as_numpy(self):
        """`SaveOutputs` serializes whatever lands in the results dict. A tensor still attached
        to the graph would carry the autograd history of the entire run into a file — which is
        why the base implementation detaches, and why this must too."""
        model = self._model()
        model._trajectories = torch.zeros(5, 7, 4, requires_grad=True) * 1.0

        traj = model.extra_outputs()["trajectories"]

        assert isinstance(traj, np.ndarray), "must survive serialization by SaveOutputs"

    def test_the_axes_are_time_first(self):
        """`(n_bins, n_trajectories, d)`, and the order is not cosmetic: `_generate_trajectories`
        integrates `odeint(network, X_0_sample, t_bins)` and torchdiffeq returns
        `(len(t), *y0.shape)`. Both parameters default to 100, so a default run cannot tell the
        two axes apart — a consumer reading shape[0] as a population size would be wrong on
        every non-default run and right on every default one."""
        model = self._model()
        model._trajectories = torch.zeros(5, 7, 4)

        traj = model.extra_outputs()["trajectories"]

        assert traj.shape[0] == model.n_bins
        assert traj.shape[1] == model.n_trajectories
