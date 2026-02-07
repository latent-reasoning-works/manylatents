# Latent ODE LightningModule Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Latent ODE as a LightningModule to manyLatents — following the existing `Reconstruction` + `Autoencoder` decomposition pattern — for learning continuous-time dynamics via an encode → ODE integrate → decode architecture.

**Architecture:** Follows the existing autoencoder pattern exactly. A `LatentODENetwork` (nn.Module) contains encoder, ODE vector field, and decoder. A `LatentODE` (LightningModule) wraps it for training with `training_step`, `configure_optimizers`, etc. Training minimizes reconstruction `||x - decode(odesolve(encode(x)))||²`. The `encode()` method returns the ODE endpoint z_T as `(n_samples, latent_dim)`, compatible with all metrics via the existing LightningModule embedding extraction in `experiment.py`.

**Tech Stack:** PyTorch, PyTorch Lightning, torchdiffeq (ODE solvers), Hydra (config)

**Template files** (the existing autoencoder pattern to follow):
- `manylatents/algorithms/lightning/reconstruction.py` — LightningModule wrapper
- `manylatents/algorithms/lightning/networks/autoencoder.py` — nn.Module network
- `manylatents/configs/algorithms/lightning/ae_reconstruction.yaml` — Hydra config

---

## File Map

| Action | Path |
|--------|------|
| Modify | `pyproject.toml` (add `torchdiffeq`, optional `torchsde`) |
| Create | `manylatents/algorithms/lightning/networks/latent_ode.py` (network) |
| Create | `manylatents/algorithms/lightning/latent_ode.py` (LightningModule) |
| Create | `manylatents/configs/algorithms/lightning/latent_ode.yaml` (Hydra config) |
| Create | `tests/algorithms/test_latent_ode.py` (tests) |

---

### Task 1: Add torchdiffeq Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add torchdiffeq to dependencies**

In `pyproject.toml`, add `torchdiffeq` to the `dependencies` list under the `# Deep learning` comment block, after `"transformers>=4.40"`:

```toml
    "torchdiffeq>=0.2",
```

Also add a `dynamics` optional dependency group for future SDE support:

```toml
[project.optional-dependencies]
slurm = ["shop"]
dynamics = ["torchsde>=0.2"]
```

**Step 2: Sync dependencies**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv sync`
Expected: Resolves and installs `torchdiffeq`. No conflicts.

**Step 3: Verify import**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from torchdiffeq import odeint; print('OK')"`
Expected: Prints `OK`.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add torchdiffeq dependency for Latent ODE"
```

---

### Task 2: Write Failing Tests

**Files:**
- Create: `tests/algorithms/test_latent_ode.py`

**Step 1: Write the test file**

```python
"""Tests for Latent ODE network and LightningModule."""

import pytest
import torch


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
        import functools

        net = LatentODENetwork(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        return LatentODE(
            network=net,
            optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
            loss=torch.nn.MSELoss(),
        )

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
        """configure_optimizers returns an optimizer."""
        model = self._make_model()
        # Need a trainer for max_epochs access in scheduler
        import lightning as L
        trainer = L.Trainer(max_epochs=3)
        model.trainer = trainer
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config

    def test_end_to_end_training(self):
        """Full Lightning training loop (few epochs, tiny data)."""
        import lightning as L
        from torch.utils.data import DataLoader, TensorDataset

        model = self._make_model(input_dim=30, latent_dim=4, hidden_dim=16)
        X = torch.randn(64, 30)
        ds = TensorDataset(X)

        # Wrap in dict-returning DataLoader (matching manyLatents batch format)
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

        # Verify embeddings work post-training
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py -v --tb=short 2>&1 | head -40`
Expected: All fail with `ModuleNotFoundError`.

**Step 3: Commit**

```bash
git add tests/algorithms/test_latent_ode.py
git commit -m "test: add failing tests for LatentODE LightningModule"
```

---

### Task 3: Implement LatentODENetwork

**Files:**
- Create: `manylatents/algorithms/lightning/networks/latent_ode.py`

**Step 1: Write the network**

Follow the `Autoencoder` pattern. Key differences: forward returns `(x_hat, z_T)` instead of just `x_hat`, and `encode()` does encode + ODE integrate (not just the encoder MLP).

```python
"""Latent ODE network: encode → ODE integrate → decode.

Network components for the Latent ODE architecture. This is the nn.Module
(the network), not the training wrapper. Analogous to how Autoencoder is
the network used by Reconstruction.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ODEFunc(nn.Module):
    """Learned vector field dz/dt = f(t, z).

    A neural network that outputs the time derivative of the latent state.
    The forward signature is f(t, z) as required by torchdiffeq.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        prev = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.Tanh()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.net(z)


class LatentODENetwork(nn.Module):
    """Complete Latent ODE: encode → integrate dz/dt = f(t,z) → decode.

    Analogous to Autoencoder but with an ODE solver between encoder and decoder.

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent state dimension (ODE state size).
        hidden_dim: Hidden width in the ODE vector field.
        encoder_hidden_dims: Hidden layer sizes for encoder MLP.
        decoder_hidden_dims: Hidden layer sizes for decoder MLP.
        ode_n_layers: Number of hidden layers in ODEFunc.
        solver: ODE solver name ('dopri5', 'euler', 'rk4', etc.).
        rtol: Relative tolerance for adaptive solvers.
        atol: Absolute tolerance for adaptive solvers.
        use_adjoint: If True, use odeint_adjoint for O(1) memory backprop.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        ode_n_layers: int = 2,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_adjoint: bool = True,
    ):
        super().__init__()
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256]

        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        # Encoder: input_dim -> latent_dim
        enc_layers = []
        prev = input_dim
        for h in encoder_hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ODE vector field
        self.ode_func = ODEFunc(latent_dim, hidden_dim, ode_n_layers)

        # Decoder: latent_dim -> input_dim
        dec_layers = []
        prev = latent_dim
        for h in decoder_hidden_dims:
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def _integrate(self, z_0: Tensor, t_span: Tensor) -> Tensor:
        """Run ODE solver. Returns trajectory (n_times, batch, latent_dim)."""
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint_fn
        else:
            from torchdiffeq import odeint as odeint_fn
        return odeint_fn(
            self.ode_func, z_0, t_span,
            method=self.solver, rtol=self.rtol, atol=self.atol,
        )

    def forward(
        self, x: Tensor, t_span: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Full forward: encode → integrate → decode.

        Args:
            x: (batch, input_dim) input data.
            t_span: Integration time points. Default [0, 1].

        Returns:
            x_hat: (batch, input_dim) reconstruction from z_T.
            z_T: (batch, latent_dim) ODE endpoint embedding.
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device)
        z_0 = self.encoder(x)
        z_traj = self._integrate(z_0, t_span)
        z_T = z_traj[-1]
        x_hat = self.decoder(z_T)
        return x_hat, z_T

    def encode(self, x: Tensor, t_span: Optional[Tensor] = None) -> Tensor:
        """Encode → integrate → return z_T. No decoder.

        This is what experiment.py calls for embedding extraction.

        Returns:
            z_T: (batch, latent_dim) ODE endpoint.
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device)
        z_0 = self.encoder(x)
        z_traj = self._integrate(z_0, t_span)
        return z_traj[-1]

    def get_latent_trajectory(
        self, x: Tensor, t_eval: Tensor
    ) -> Tensor:
        """Full trajectory for Geomancy per-timepoint geometric profiling.

        Args:
            x: (batch, input_dim) input data.
            t_eval: (n_times,) evaluation timepoints.

        Returns:
            z_traj: (n_times, batch, latent_dim) trajectory.
        """
        z_0 = self.encoder(x)
        return self._integrate(z_0, t_eval)
```

**Step 2: Run network tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py::TestLatentODENetwork -v --tb=short`
Expected: All 5 tests PASS.

**Step 3: Commit**

```bash
git add manylatents/algorithms/lightning/networks/latent_ode.py
git commit -m "feat: add LatentODENetwork (encode → ODE → decode)"
```

---

### Task 4: Implement LatentODE LightningModule

**Files:**
- Create: `manylatents/algorithms/lightning/latent_ode.py`

**Step 1: Write the LightningModule**

Follow `Reconstruction` pattern. Key differences:
- `shared_step` unpacks `(x_hat, z_T)` from network forward (single pass, no redundant encoder call)
- `encode()` delegates to `self.network.encode(x, t_span)` which returns z_T
- Integration times are stored on the module and passed through
- Loss receives `outputs=x_hat, targets=x` — compatible with existing MSELoss

```python
"""Latent ODE LightningModule training wrapper.

Follows the Reconstruction pattern: wraps a LatentODENetwork for training
with Lightning's training loop, checkpointing, gradient clipping, etc.
"""

import functools
import logging

import hydra_zen
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LatentODE(LightningModule):
    """Lightning training wrapper for Latent ODE.

    Follows the same pattern as Reconstruction. Trains an encoder-ODE-decoder
    architecture with reconstruction loss.

    Args:
        datamodule: Data module for loading train/val/test data.
        network: Hydra config or instantiated LatentODENetwork.
        loss: Hydra config or instantiated loss module.
        optimizer: Hydra config for optimizer (partial instantiation).
        init_seed: Random seed for weight initialization.
        integration_times: ODE integration time span [t_0, t_T].
    """

    def __init__(
        self,
        network,
        optimizer,
        loss,
        datamodule=None,
        init_seed: int = 42,
        integration_times: list[float] | None = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.loss_config = loss
        self.init_seed = init_seed
        self.integration_times = integration_times or [0.0, 1.0]

        self.save_hyperparameters(ignore=["datamodule"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        """Infer input_dim from data if needed, then build network."""
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None and self.datamodule is not None:
                first_batch = next(iter(self.datamodule.train_dataloader()))
                data = first_batch["data"] if isinstance(first_batch, dict) else first_batch[0]
                self.network_config["input_dim"] = data.shape[1]
            self.configure_model()
        elif self.network is None:
            # Already an instantiated nn.Module (e.g., in tests)
            self.network = self.network_config
            self.network_config = None
            self.loss_fn = self.loss_config if isinstance(self.loss_config, nn.Module) else hydra_zen.instantiate(self.loss_config)
            self._optimizer_partial = self.optimizer_config

    def configure_model(self):
        """Instantiate network and loss from Hydra configs."""
        torch.manual_seed(self.init_seed)

        cfg_map = {
            "network": self.network_config,
            "loss_fn": self.loss_config,
        }
        for attr, cfg in cfg_map.items():
            if isinstance(cfg, (dict, DictConfig)):
                setattr(self, attr, hydra_zen.instantiate(cfg))
            else:
                setattr(self, attr, cfg)

        self._optimizer_partial = self.optimizer_config
        logger.info(
            f"Instantiated network={self.network.__class__.__name__}, "
            f"loss_fn={self.loss_fn.__class__.__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns reconstruction x_hat."""
        assert self.network is not None, "Network not configured. Call setup() first."
        t_span = torch.tensor(self.integration_times, device=x.device)
        x_hat, _z_T = self.network(x, t_span)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns z_T embedding. Called by experiment.py for metrics."""
        assert self.network is not None, "Network not configured. Call setup() first."
        t_span = torch.tensor(self.integration_times, device=x.device)
        return self.network.encode(x, t_span)

    def shared_step(self, batch, batch_idx, phase: str) -> dict:
        x = batch["data"] if isinstance(batch, dict) else batch[0]
        t_span = torch.tensor(self.integration_times, device=x.device)
        x_hat, z_T = self.network(x, t_span)

        # Loss: compatible with MSELoss(outputs, targets, **kwargs)
        if hasattr(self.loss_fn, "components"):
            extras = {"latent": z_T, "raw": x}
            comps = self.loss_fn.components(outputs=x_hat, targets=x, **extras)
            loss = sum(comps.values())
            self.log_dict(
                {f"{phase}_{k}": v for k, v in comps.items()},
                on_step=False, on_epoch=True, prog_bar=False,
            )
        else:
            loss = self.loss_fn(outputs=x_hat, targets=x)

        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_z_norm", z_T.norm(dim=-1).mean(), on_step=False, on_epoch=True)
        return {"loss": loss, "outputs": x_hat}

    def training_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, phase="test")
        self.log("test_loss", out["loss"], prog_bar=True, on_epoch=True)
        return out

    def configure_optimizers(self):
        """Instantiate optimizer, add cosine annealing scheduler."""
        if isinstance(self._optimizer_partial, functools.partial):
            optimizer = self._optimizer_partial(self.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self._optimizer_partial)
            optimizer = optimizer_partial(self.parameters())

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
```

**Step 2: Run LightningModule tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py::TestLatentODELightningModule -v --tb=short`
Expected: All 5 tests PASS (including end-to-end training).

**Step 3: Run all tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py -v --tb=short`
Expected: All 11 tests PASS.

**Step 4: Commit**

```bash
git add manylatents/algorithms/lightning/latent_ode.py
git commit -m "feat: add LatentODE LightningModule training wrapper"
```

---

### Task 5: Add Hydra Config

**Files:**
- Create: `manylatents/configs/algorithms/lightning/latent_ode.yaml`

**Step 1: Create the config**

Follow `ae_reconstruction.yaml` pattern exactly:

```yaml
_target_: manylatents.algorithms.lightning.latent_ode.LatentODE
datamodule: ${data}
network:
  _target_: manylatents.algorithms.lightning.networks.latent_ode.LatentODENetwork
  input_dim: null
  latent_dim: 16
  hidden_dim: 128
  encoder_hidden_dims: [256, 128]
  decoder_hidden_dims: [128, 256]
  ode_n_layers: 2
  solver: dopri5
  rtol: 1e-4
  atol: 1e-4
  use_adjoint: true
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
loss:
  _target_: manylatents.algorithms.lightning.losses.mse.MSELoss
integration_times: [0.0, 1.0]
```

**Step 2: Verify Hydra resolution**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -m manylatents.main algorithms/lightning=latent_ode data=swissroll logger=none trainer=default --cfg job 2>&1 | head -40`
Expected: Prints resolved config with the latent_ode settings.

**Step 3: Run Hydra instantiation test**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py::TestLatentODEHydra -v --tb=short`
Expected: PASS.

**Step 4: Commit**

```bash
git add manylatents/configs/algorithms/lightning/latent_ode.yaml
git commit -m "feat: add Hydra config for Latent ODE"
```

---

### Task 6: End-to-End Pipeline Verification

**Step 1: Run through the standard pipeline**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -m manylatents.main algorithms/lightning=latent_ode data=swissroll metrics=test_metric logger=none trainer=default trainer.max_epochs=3`

Expected: Completes without error. Pipeline should:
1. Instantiate SwissRoll datamodule
2. Instantiate LatentODE (input_dim auto-detected in setup())
3. Train with Lightning Trainer (3 epochs)
4. Extract embeddings via `algorithm.encode(test_tensor)`
5. Compute test metrics

**Step 2: If pipeline issues arise**

Check `experiment.py` LightningModule path at lines 358-395. It does:
```python
trainer.fit(algorithm, datamodule=datamodule)
if hasattr(algorithm, "encode"):
    latents = algorithm.encode(test_tensor)
```

Our `LatentODE.encode(test_tensor)` returns `z_T` of shape `(n_samples, latent_dim)` — this should work.

If the batch format doesn't match (dict vs tuple), fix in `latent_ode.py`'s `shared_step` — it already handles both formats.

**Step 3: Commit if fixes were needed**

```bash
git add manylatents/algorithms/lightning/latent_ode.py
git commit -m "fix: address pipeline integration issues in LatentODE"
```

---

### Task 7: Final Cleanup

**Step 1: Run full test suite**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/algorithms/test_latent_ode.py -v`
Expected: All tests PASS.

**Step 2: Run pre-commit**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pre-commit run --files manylatents/algorithms/lightning/latent_ode.py manylatents/algorithms/lightning/networks/latent_ode.py manylatents/configs/algorithms/lightning/latent_ode.yaml tests/algorithms/test_latent_ode.py pyproject.toml`
Expected: All pass. Fix any formatting issues.

**Step 3: Commit if needed**

```bash
git add -u
git commit -m "style: apply pre-commit formatting to latent_ode files"
```

---

## Constraints Recap

- **Follow the Reconstruction + Autoencoder decomposition** — network is nn.Module, training wrapper is LightningModule
- **`encode()` must return `(n_samples, latent_dim)`** z_T — compatible with experiment.py embedding extraction
- **Gradient clipping** — Trainer config already has `gradient_clip_val: 1.0`, critical for ODE stability
- **`use_adjoint=True` by default** — O(1) memory backprop through ODE solver
- **Batch format**: `batch["data"]` (dict) matching existing data pipeline, with fallback to `batch[0]` (tuple)
- **Do NOT modify** Reconstruction, Autoencoder, LatentModule, or experiment.py
- **No SDE, Graph ODE, or trajectory metrics** — future follow-ups
- Tests must run in seconds (small data, few epochs)
