"""Tests for Latent ODE network and LightningModule."""

import functools

import pytest
import torch

torchdiffeq = pytest.importorskip("torchdiffeq")


class TestLatentODENetwork:
    """Tests for the LatentODENetwork nn.Module."""

    def test_forward_shapes(self):
        """forward returns (x_hat, z_T) with correct shapes."""
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        net = LatentODENetwork(input_dim=50, latent_dim=8, hidden_dim=32)
        x = torch.randn(16, 50)
        x_hat, z_T = net(x)
        assert x_hat.shape == (16, 50)
        assert z_T.shape == (16, 8)

    def test_encode_returns_z_T(self):
        """encode returns ODE endpoint z_T, not z_0."""
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        net = LatentODENetwork(input_dim=50, latent_dim=8, hidden_dim=32)
        x = torch.randn(16, 50)
        z_T = net.encode(x)
        assert z_T.shape == (16, 8)

    def test_get_latent_trajectory(self):
        """get_latent_trajectory returns (n_times, batch, latent_dim)."""
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        net = LatentODENetwork(input_dim=50, latent_dim=8, hidden_dim=32)
        x = torch.randn(16, 50)
        t_eval = torch.linspace(0, 1, 10)
        z_traj = net.get_latent_trajectory(x, t_eval)
        assert z_traj.shape == (10, 16, 8)

    def test_output_is_finite(self):
        """No NaN or Inf in outputs."""
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        net = LatentODENetwork(input_dim=30, latent_dim=4, hidden_dim=16)
        x = torch.randn(8, 30)
        x_hat, z_T = net(x)
        assert torch.isfinite(x_hat).all()
        assert torch.isfinite(z_T).all()

    def test_adjoint_vs_standard(self):
        """Both adjoint and standard solvers produce same shapes."""
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        x = torch.randn(8, 20)
        net_adj = LatentODENetwork(input_dim=20, latent_dim=4, hidden_dim=16, use_adjoint=True)
        net_std = LatentODENetwork(input_dim=20, latent_dim=4, hidden_dim=16, use_adjoint=False)
        x_hat_a, z_T_a = net_adj(x)
        x_hat_s, z_T_s = net_std(x)
        assert x_hat_a.shape == x_hat_s.shape
        assert z_T_a.shape == z_T_s.shape


class TestLatentODELightningModule:
    """Tests for the LatentODE LightningModule wrapper."""

    def _make_model(self, input_dim=50, latent_dim=8, hidden_dim=32):
        from manylatents.algorithms.lightning.latent_ode import LatentODE
        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        from manylatents.algorithms.lightning.losses.mse import MSELoss

        net = LatentODENetwork(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model = LatentODE(
            network=net,
            optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
            loss=MSELoss(),
        )
        model.setup()
        return model

    def test_training_step(self):
        """Single training step runs and produces valid loss."""
        model = self._make_model()
        batch = {"data": torch.randn(16, 50)}
        result = model.training_step(batch, 0)
        loss = result["loss"]
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_validation_step(self):
        """Validation step runs without error."""
        model = self._make_model()
        batch = {"data": torch.randn(16, 50)}
        result = model.validation_step(batch, 0)
        assert torch.isfinite(result["loss"])

    def test_encode(self):
        """encode returns (batch, latent_dim) — used by experiment.py for metrics."""
        model = self._make_model()
        x = torch.randn(32, 50)
        z = model.encode(x)
        assert z.shape == (32, 8)

    def test_forward(self):
        """forward returns x_hat reconstruction."""
        model = self._make_model()
        x = torch.randn(16, 50)
        x_hat = model(x)
        assert x_hat.shape == (16, 50)

    def test_configure_optimizers(self):
        """configure_optimizers returns an optimizer dict."""
        import lightning as L

        model = self._make_model()
        model.trainer = L.Trainer(max_epochs=3)
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config

    def test_end_to_end_training(self):
        """Full Lightning training loop (few epochs, tiny data)."""
        import lightning as L
        from torch.utils.data import DataLoader

        model = self._make_model(input_dim=30, latent_dim=4, hidden_dim=16)
        X = torch.randn(64, 30)

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, tensor):
                self.tensor = tensor

            def __len__(self):
                return len(self.tensor)

            def __getitem__(self, idx):
                return {"data": self.tensor[idx]}

        loader = DataLoader(DictDataset(X), batch_size=16)
        trainer = L.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            enable_progress_bar=False,
            gradient_clip_val=1.0,
            logger=False,
        )
        trainer.fit(model, loader)

        z = model.encode(X)
        assert z.shape == (64, 4)
        assert torch.isfinite(z).all()


class TestLatentODEHydra:
    """Hydra instantiation tests."""

    def test_instantiate_network_from_dict(self):
        """Hydra can instantiate LatentODENetwork from config dict."""
        from hydra.utils import instantiate

        from manylatents.algorithms.lightning.networks.latent_ode import LatentODENetwork

        cfg = {
            "_target_": "manylatents.algorithms.lightning.networks.latent_ode.LatentODENetwork",
            "input_dim": 50,
            "latent_dim": 8,
            "hidden_dim": 32,
        }
        net = instantiate(cfg)
        assert isinstance(net, LatentODENetwork)
        assert net.latent_dim == 8
