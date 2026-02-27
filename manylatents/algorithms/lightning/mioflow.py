"""MIOFlow LightningModule training wrapper.

Manifold Interpolating Optimal-Transport Flows for trajectory inference.
Uses manual optimization to iterate over time intervals within each training step.

Reference: Huguet et al., arXiv:2206.14928 (2022)
"""

import functools
import logging

import hydra_zen
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class MIOFlow(LightningModule):
    """Lightning training wrapper for MIOFlow trajectory inference.

    Trains a Neural ODE velocity field using Optimal Transport losses over
    time-labeled populations. Supports 3-phase training: local pretrain,
    global, and local finetune.

    Args:
        network: Hydra config or instantiated MIOFlowODEFunc.
        optimizer: Hydra config for optimizer (partial instantiation).
        datamodule: Data module for loading time-labeled data.
        init_seed: Random seed for weight initialization.
        n_local_epochs: Epochs of local (per-interval) pre-training.
        n_global_epochs: Epochs of global (full-trajectory) training.
        n_post_local_epochs: Epochs of local fine-tuning after global.
        lambda_ot: Weight for OT loss.
        lambda_energy: Weight for energy regularisation.
        lambda_density: Weight for density loss.
        energy_time_steps: Sub-steps for energy loss computation.
        sample_size: Points sampled per time step (None = use all).
        n_trajectories: Number of trajectories to generate after training.
        n_bins: Number of time bins for trajectory integration.
    """

    def __init__(
        self,
        network,
        optimizer,
        datamodule=None,
        init_seed: int = 42,
        # Training phases
        n_local_epochs: int = 0,
        n_global_epochs: int = 100,
        n_post_local_epochs: int = 0,
        # Loss weights
        lambda_ot: float = 1.0,
        lambda_energy: float = 0.01,
        lambda_density: float = 0.0,
        energy_time_steps: int = 10,
        # Data
        sample_size: int | None = None,
        # Output
        n_trajectories: int = 100,
        n_bins: int = 100,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Training phases
        self.n_local_epochs = n_local_epochs
        self.n_global_epochs = n_global_epochs
        self.n_post_local_epochs = n_post_local_epochs

        # Loss weights
        self.lambda_ot = lambda_ot
        self.lambda_energy = lambda_energy
        self.lambda_density = lambda_density
        self.energy_time_steps = energy_time_steps

        # Data
        self.sample_size = sample_size

        # Output
        self.n_trajectories = n_trajectories
        self.n_bins = n_bins

        self.save_hyperparameters(ignore=["datamodule", "network"])
        self.network: nn.Module | None = None
        self._trajectories: Tensor | None = None

    @property
    def total_epochs(self) -> int:
        return self.n_local_epochs + self.n_global_epochs + self.n_post_local_epochs

    def _get_training_mode(self, epoch: int) -> str:
        """Determine training mode based on current epoch."""
        if epoch < self.n_local_epochs:
            return "local"
        elif epoch < self.n_local_epochs + self.n_global_epochs:
            return "global"
        else:
            return "local"

    def setup(self, stage=None):
        """Infer input_dim from data if needed, then build network."""
        if self.network is not None:
            return
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None and self.datamodule is not None:
                first_batch = next(iter(self.datamodule.train_dataloader()))
                data = first_batch["data"] if isinstance(first_batch, dict) else first_batch[0]
                self.network_config["input_dim"] = data.shape[1]
        self.configure_model()

    def configure_model(self):
        """Instantiate network from Hydra config."""
        torch.manual_seed(self.init_seed)
        if isinstance(self.network_config, (dict, DictConfig)):
            self.network = hydra_zen.instantiate(self.network_config)
        else:
            self.network = self.network_config
        logger.info(f"MIOFlow network: {self.network.__class__.__name__}")

    def _group_by_time(self, batch: dict) -> list[tuple[Tensor, float]]:
        """Group batch data by time labels into sorted list of (X_t, t)."""
        data = batch["data"] if isinstance(batch, dict) else batch[0]
        labels = batch["labels"] if isinstance(batch, dict) else batch[1]
        unique_times = torch.unique(labels, sorted=True)
        groups = []
        for t in unique_times:
            mask = labels == t
            groups.append((data[mask], t.item()))
        return groups

    def _local_step(self, time_groups: list[tuple[Tensor, float]]) -> dict[str, Tensor]:
        """Train on each consecutive time interval independently."""
        from torchdiffeq import odeint

        from .networks.mioflow_net import (
            mioflow_density_loss,
            mioflow_energy_loss,
            mioflow_ot_loss,
        )

        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        ot_sum = torch.tensor(0.0, device=device)
        energy_sum = torch.tensor(0.0, device=device)
        density_sum = torch.tensor(0.0, device=device)
        n_intervals = 0

        for i in range(len(time_groups) - 1):
            X_start, t_start = time_groups[i]
            X_end, t_end = time_groups[i + 1]
            X_start = X_start.to(device)
            X_end = X_end.to(device)

            # Subsample if needed
            if self.sample_size is not None:
                n = min(X_start.size(0), X_end.size(0), self.sample_size)
                X_start = X_start[torch.randperm(X_start.size(0))[:n]]
                X_end = X_end[torch.randperm(X_end.size(0))[:n]]

            t_interval = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
            X_pred = odeint(self.network, X_start, t_interval)[1]

            interval_loss = torch.tensor(0.0, device=device)

            if self.lambda_ot > 0:
                ot_v = mioflow_ot_loss(X_pred, X_end)
                interval_loss = interval_loss + self.lambda_ot * ot_v
                ot_sum = ot_sum + ot_v.detach()

            if self.lambda_density > 0:
                den_v = mioflow_density_loss(X_pred, X_end)
                interval_loss = interval_loss + self.lambda_density * den_v
                density_sum = density_sum + den_v.detach()

            if self.lambda_energy > 0:
                e_t = torch.linspace(t_start, t_end, self.energy_time_steps, device=device)
                eng_v = mioflow_energy_loss(self.network, X_start, e_t)
                interval_loss = interval_loss + self.lambda_energy * eng_v
                energy_sum = energy_sum + eng_v.detach()

            total_loss = total_loss + interval_loss
            n_intervals += 1

        if n_intervals > 0:
            total_loss = total_loss / n_intervals

        return {
            "loss": total_loss,
            "ot_loss": ot_sum / max(n_intervals, 1),
            "energy_loss": energy_sum / max(n_intervals, 1),
            "density_loss": density_sum / max(n_intervals, 1),
        }

    def _global_step(self, time_groups: list[tuple[Tensor, float]]) -> dict[str, Tensor]:
        """Train on full trajectory end-to-end."""
        from torchdiffeq import odeint

        from .networks.mioflow_net import (
            mioflow_density_loss,
            mioflow_energy_loss,
            mioflow_ot_loss,
        )

        device = self.device
        X_0 = time_groups[0][0].to(device)
        times = [t for _, t in time_groups]
        t_seq = torch.tensor(times, device=device, dtype=torch.float32)

        if self.sample_size is not None and X_0.size(0) > self.sample_size:
            idx = torch.randperm(X_0.size(0))[: self.sample_size]
            X_0 = X_0[idx]

        trajectory = odeint(self.network, X_0, t_seq)

        total_loss = torch.tensor(0.0, device=device)
        ot_sum = torch.tensor(0.0, device=device)
        density_sum = torch.tensor(0.0, device=device)

        for i in range(1, len(t_seq)):
            X_pred = trajectory[i]
            X_true = time_groups[i][0].to(device)

            if self.sample_size is not None and X_true.size(0) > self.sample_size:
                X_true = X_true[torch.randperm(X_true.size(0))[: self.sample_size]]

            if self.lambda_ot > 0:
                ot_v = mioflow_ot_loss(X_pred, X_true)
                total_loss = total_loss + self.lambda_ot * ot_v
                ot_sum = ot_sum + ot_v.detach()

            if self.lambda_density > 0:
                den_v = mioflow_density_loss(X_pred, X_true)
                total_loss = total_loss + self.lambda_density * den_v
                density_sum = density_sum + den_v.detach()

        energy_v = torch.tensor(0.0, device=device)
        if self.lambda_energy > 0:
            e_t = torch.linspace(times[0], times[-1], self.energy_time_steps, device=device)
            energy_v = mioflow_energy_loss(self.network, X_0, e_t)
            total_loss = total_loss + self.lambda_energy * energy_v

        return {
            "loss": total_loss,
            "ot_loss": ot_sum / max(len(t_seq) - 1, 1),
            "energy_loss": energy_v.detach(),
            "density_loss": density_sum / max(len(t_seq) - 1, 1),
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        mode = self._get_training_mode(self.current_epoch)
        time_groups = self._group_by_time(batch)

        if len(time_groups) < 2:
            return  # Need at least 2 time points

        if mode == "local":
            result = self._local_step(time_groups)
        else:
            result = self._global_step(time_groups)

        opt.zero_grad()
        self.manual_backward(result["loss"])
        opt.step()

        self.log("train_loss", result["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_ot_loss", result["ot_loss"], on_step=False, on_epoch=True)
        self.log("train_energy_loss", result["energy_loss"], on_step=False, on_epoch=True)
        self.log("train_density_loss", result["density_loss"], on_step=False, on_epoch=True)
        self.log("training_mode", float(0 if mode == "local" else 1), on_step=False, on_epoch=True)

    def on_train_end(self):
        """Generate trajectories after training completes."""
        self._generate_trajectories()

    def _generate_trajectories(self):
        """Integrate n_trajectories paths over n_bins time steps."""
        from torchdiffeq import odeint

        if self.datamodule is None:
            return

        # Get all data grouped by time
        all_data = []
        for batch in self.datamodule.train_dataloader():
            groups = self._group_by_time(batch)
            all_data.extend(groups)

        if not all_data:
            return

        # Sort by time and deduplicate
        all_data.sort(key=lambda x: x[1])
        times = sorted(set(t for _, t in all_data))

        # Get initial conditions from earliest time
        X_0 = all_data[0][0].to(self.device)
        n = min(self.n_trajectories, X_0.size(0))
        idx = torch.randperm(X_0.size(0))[:n]
        X_0_sample = X_0[idx]

        t_bins = torch.linspace(min(times), max(times), self.n_bins, device=self.device)

        self.network.eval()
        with torch.no_grad():
            self._trajectories = odeint(self.network, X_0_sample, t_bins)
        self.network.train()
        logger.info(f"Trajectories generated: shape={self._trajectories.shape}")

    def encode(self, x: Tensor) -> Tensor:
        """Integrate x from t=0 to t=1, return endpoint positions.

        This produces (n, d) embeddings compatible with the pipeline.
        """
        from torchdiffeq import odeint

        assert self.network is not None, "Network not configured. Call setup() first."

        # Get time range from data if available
        t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=torch.float32)
        if self.datamodule is not None:
            try:
                batch = next(iter(self.datamodule.train_dataloader()))
                groups = self._group_by_time(batch)
                if len(groups) >= 2:
                    times = [t for _, t in groups]
                    t_span = torch.tensor(
                        [min(times), max(times)], device=x.device, dtype=torch.float32
                    )
            except StopIteration:
                pass

        self.network.eval()
        with torch.no_grad():
            trajectory = odeint(self.network, x, t_span)
        return trajectory[-1]  # endpoint positions

    @property
    def trajectories(self) -> Tensor | None:
        """Full trajectories (n_bins, n_traj, d) if generated."""
        return self._trajectories

    def test_step(self, batch, batch_idx):
        time_groups = self._group_by_time(batch)
        if len(time_groups) < 2:
            return
        result = self._global_step(time_groups)
        self.log("test_loss", result["loss"], prog_bar=True, on_epoch=True)
        return result

    def configure_optimizers(self):
        """Instantiate optimizer."""
        if isinstance(self.optimizer_config, functools.partial):
            optimizer = self.optimizer_config(self.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
            optimizer = optimizer_partial(self.parameters())
        return optimizer
