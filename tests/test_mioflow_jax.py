"""Tests for JAX MIOFlow LatentModule."""

import pytest
import numpy as np
import torch

jax = pytest.importorskip("jax")
diffrax = pytest.importorskip("diffrax")
optax = pytest.importorskip("optax")


class TestMIOFlowJAX:
    """Tests for the JAX MIOFlow LatentModule."""

    @pytest.fixture
    def time_labeled_data(self):
        """Create time-labeled synthetic data (3 time points)."""
        torch.manual_seed(42)
        n_per_time = 20
        dim = 5
        data_parts = []
        label_parts = []
        for t in [0.0, 0.5, 1.0]:
            data_parts.append(torch.randn(n_per_time, dim) + t)
            label_parts.append(torch.full((n_per_time,), t))
        return torch.cat(data_parts), torch.cat(label_parts)

    @pytest.fixture
    def module(self):
        from manylatents.algorithms.latent.mioflow_jax import MIOFlowJAX

        return MIOFlowJAX(
            hidden_dim=16,
            n_epochs=5,  # few epochs for fast tests
            lambda_ot=1.0,
            lambda_energy=0.01,
            energy_time_steps=5,
            n_trajectories=10,
            n_bins=10,
        )

    def test_is_latent_module(self, module):
        from manylatents.algorithms.latent.latent_module_base import LatentModule

        assert isinstance(module, LatentModule)

    def test_fit_requires_labels(self, module, time_labeled_data):
        x, y = time_labeled_data
        with pytest.raises(ValueError, match="requires time labels"):
            module.fit(x, y=None)

    def test_fit_runs(self, module, time_labeled_data):
        x, y = time_labeled_data
        module.fit(x, y)
        assert module._is_fitted

    def test_transform_shape(self, module, time_labeled_data):
        x, y = time_labeled_data
        module.fit(x, y)
        result = module.transform(x)
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    def test_fit_transform(self, module, time_labeled_data):
        x, y = time_labeled_data
        result = module.fit_transform(x, y)
        assert result.shape == x.shape

    def test_trajectories_generated(self, module, time_labeled_data):
        x, y = time_labeled_data
        module.fit(x, y)
        traj = module.trajectories
        assert traj is not None
        assert traj.shape[0] == 10  # n_bins
        assert traj.shape[2] == 5  # dim

    def test_deterministic(self, time_labeled_data):
        """Same seed produces same output."""
        from manylatents.algorithms.latent.mioflow_jax import MIOFlowJAX

        x, y = time_labeled_data

        m1 = MIOFlowJAX(
            hidden_dim=16, n_epochs=3, init_seed=42, n_bins=5, n_trajectories=5
        )
        m1.fit(x, y)
        r1 = m1.transform(x[:5])

        m2 = MIOFlowJAX(
            hidden_dim=16, n_epochs=3, init_seed=42, n_bins=5, n_trajectories=5
        )
        m2.fit(x, y)
        r2 = m2.transform(x[:5])

        np.testing.assert_allclose(r1.numpy(), r2.numpy(), rtol=1e-4)
