"""Cflows LightningModule — MIOFlow-family neural-ODE trajectory model.

Trains a time-conditioned flow that pushes a point cloud from one measured
timepoint to the next, matching the predicted cloud to the observed cloud with
an optimal-transport (OT) point-cloud loss. This is the *v0 / LOCAL* regime of
the MIOFlow family (Huguet et al., arXiv:2206.14928): the loss is summed over
consecutive timepoint pairs, each pair trained on the real cells at ``t_k`` as
the initial condition.

Surface mirrors :class:`~manylatents.algorithms.lightning.latent_ode.LatentODE`
(``__init__(network, optimizer, loss, datamodule, ...)``, ``setup``,
``configure_model``, ``configure_optimizers``, ``encode``) so it slots into the
same experiment / api.run machinery. Unlike ``LatentODE`` — which reads only
``batch["data"]`` and reconstructs — ``Cflows`` also reads ``batch["time"]``
(per-cell timepoint labels) and trains a *trajectory*, not a reconstruction.

Integration space (design decision, documented per the task):
    We reuse the whole :class:`LatentODENetwork` (``encode -> integrate in the
    latent -> decode``). The ODE is integrated **in the latent space** but the
    OT / density match is done **in data space** against the real cells, via the
    decoder. Concretely, for a pair ``(t_k, t_{k+1})`` we call
    ``x_pred, z_T = network(x_at_t_k, t_span=[t_k, t_{k+1}])`` and match
    ``x_pred`` (decoded prediction at ``t_{k+1}``) to the observed cloud.

    Why not the two "pure" options?
      * *Pure latent-space OT match* (integrate + match in the latent) is the
        geodesic-autoencoder regime. Without a geometry-preserving pretraining
        the encoder is free to collapse every cell to one point (OT loss -> 0,
        degenerate). The geodesic-AE latent is **DEFERRED** (see below), so this
        is not a coherent v0.
      * *Pure data-space integration* can't collapse but under-uses the network
        (only ``ODEFunc``) and yields no genuine latent embedding for
        ``encode()``.
    Matching in data space against fixed real cells cannot collapse (the target
    is not learnable), and ``encode()`` still returns a real latent ``z_T``.
    With a near-lossless AE (``latent_dim >= input_dim``) the flow recovers a
    known drift; this is what ``tests/test_cflows_model.py`` proves.

Time enters via ``batch["time"]``: cells are grouped by unique sorted timepoint,
and each consecutive pair's ``t_span`` is built from those *actual* timepoint
values (not a global constant) and handed to the ODE solver.

DEFERRED (intentionally NOT built here — clear extension points):
    * Geodesic-autoencoder latent (geometry-preserving encoder pretraining).
    * Growth-rate / unbalanced-OT (cell birth-death, non-uniform marginals).
    * The GLOBAL regime (integrate the full trajectory end-to-end in one shot).
    * The decoder + Granger GRN head (a separate task).
    ``extra_outputs()`` returns ``{}`` for now.
"""

import functools
import logging

import hydra_zen
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor

from manylatents.algorithms.lightning.losses.cflows import (
    DensityLoss,
    OTLoss,
    energy,
)

logger = logging.getLogger(__name__)


class Cflows(LightningModule):
    """Lightning training wrapper for the Cflows trajectory flow (v0 / LOCAL).

    Args:
        network: Hydra config or instantiated ``LatentODENetwork`` (encode ->
            integrate in latent -> decode).
        optimizer: Hydra config for optimizer (partial instantiation) or a
            ``functools.partial``.
        loss: Hydra config or instantiated OT loss (``OTLoss``). This is the
            primary point-cloud loss and Cflows owns it; the density loss is
            built internally (``DensityLoss``) and the energy penalty uses the
            ``energy`` helper. The config ``loss`` node therefore carries the
            OT loss specifically.
        datamodule: Data module yielding batches with ``"data"`` and ``"time"``.
        init_seed: Random seed for weight initialization.
        integration_times: Time span used by ``encode()`` for the embedding
            (mirrors ``LatentODE``). The *training* time spans come from the
            data's timepoints, not this value.
        lambda_density: Weight on the density loss (default ``1.0``).
        lambda_energy: Weight on the trajectory-energy penalty (default ``0.0``
            for the v0 unit test; tunable).
        energy_steps: Sub-steps used to accumulate per-step velocity norms for
            the energy penalty (only evaluated when ``lambda_energy > 0``).
        density_top_k: ``top_k`` neighbours for the density loss.
    """

    def __init__(
        self,
        network,
        optimizer,
        loss,
        datamodule=None,
        init_seed: int = 42,
        integration_times: list[float] | None = None,
        lambda_density: float = 1.0,
        lambda_energy: float = 0.0,
        energy_steps: int = 10,
        density_top_k: int = 5,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.loss_config = loss
        self.init_seed = init_seed
        self.integration_times = integration_times or [0.0, 1.0]
        self.lambda_density = lambda_density
        self.lambda_energy = lambda_energy
        self.energy_steps = energy_steps
        self.density_top_k = density_top_k

        self.save_hyperparameters(ignore=["datamodule", "network", "loss"])
        self.network: nn.Module | None = None
        self.ot_loss: nn.Module | None = None
        self.density_loss: nn.Module | None = None

    # ------------------------------------------------------------------ #
    # Setup / construction (mirrors LatentODE)
    # ------------------------------------------------------------------ #
    def setup(self, stage=None):
        """Infer ``input_dim`` from data if needed, then build the network."""
        if self.network is not None:
            return  # Already configured (idempotent)
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None and self.datamodule is not None:
                first_batch = next(iter(self.datamodule.train_dataloader()))
                data = first_batch["data"] if isinstance(first_batch, dict) else first_batch[0]
                self.network_config["input_dim"] = data.shape[1]
        self.configure_model()

    def configure_model(self):
        """Instantiate network + losses from Hydra configs (or pass instances)."""
        torch.manual_seed(self.init_seed)

        if isinstance(self.network_config, (dict, DictConfig)):
            self.network = hydra_zen.instantiate(self.network_config)
        else:
            self.network = self.network_config

        # Cflows owns its losses. The config `loss` node is the OT loss; the
        # density loss and energy penalty are built/used internally.
        if isinstance(self.loss_config, (dict, DictConfig)):
            self.ot_loss = hydra_zen.instantiate(self.loss_config)
        else:
            self.ot_loss = self.loss_config if self.loss_config is not None else OTLoss()
        self.density_loss = DensityLoss()

        self._optimizer_partial = self.optimizer_config
        logger.info(
            f"Instantiated network={self.network.__class__.__name__}, "
            f"ot_loss={self.ot_loss.__class__.__name__}, "
            f"lambda_density={self.lambda_density}, lambda_energy={self.lambda_energy}"
        )

    # ------------------------------------------------------------------ #
    # Time grouping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_by_time(x: Tensor, t: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Group rows of ``x`` by unique sorted timepoint in ``t``.

        Returns a list of ``(cloud, time_scalar)`` ordered by increasing time.
        """
        unique_times = torch.unique(t, sorted=True)
        return [(x[t == tv], tv) for tv in unique_times]

    def _step_norms(self, x0: Tensor, t0: Tensor, t1: Tensor) -> list[Tensor]:
        """Per-step squared velocity norms ``||f(t, z_t)||^2`` along the latent
        trajectory from ``t0`` to ``t1`` (for the energy penalty)."""
        t_eval = torch.linspace(
            float(t0), float(t1), self.energy_steps, device=x0.device, dtype=x0.dtype
        )
        z_traj = self.network.get_latent_trajectory(x0, t_eval)  # (steps, batch, latent)
        norms: list[Tensor] = []
        for i in range(z_traj.shape[0]):
            v = self.network.ode_func(t_eval[i], z_traj[i])  # (batch, latent)
            norms.append((v**2).sum(dim=-1))  # per-sample squared norm
        return norms

    # ------------------------------------------------------------------ #
    # Core step
    # ------------------------------------------------------------------ #
    def shared_step(self, batch, batch_idx, phase: str) -> dict:
        x = batch["data"] if isinstance(batch, dict) else batch[0]
        if isinstance(batch, dict):
            t = batch.get("time")
            if t is None:
                raise KeyError(
                    "Cflows requires per-cell timepoints under batch['time']; "
                    f"got keys {sorted(batch.keys())}. Pass `time=...` to the "
                    "datamodule / api.run."
                )
        else:
            t = batch[1]

        groups = self._group_by_time(x, t)
        device = x.device

        if len(groups) < 2:
            # Nothing to flow between; emit a zero loss so the epoch is valid.
            loss = torch.zeros((), device=device, requires_grad=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return {"loss": loss}

        total = torch.zeros((), device=device)
        ot_sum = torch.zeros((), device=device)
        den_sum = torch.zeros((), device=device)
        eng_sum = torch.zeros((), device=device)

        # LOCAL regime: sum over consecutive timepoint pairs.
        for i in range(len(groups) - 1):
            x0, t0 = groups[i]
            x1, t1 = groups[i + 1]
            # t_span from the ACTUAL timepoints (not a global constant).
            t_span = torch.stack([t0, t1]).to(device=device, dtype=x.dtype)

            # encode(x0) -> integrate latent over [t_k, t_{k+1}] -> decode.
            x_pred, _z_T = self.network(x0, t_span)

            ot_v = self.ot_loss(x_pred, x1)
            pair_loss = ot_v
            ot_sum = ot_sum + ot_v.detach()

            if self.lambda_density > 0:
                top_k = min(self.density_top_k, x1.shape[0])
                den_v = self.density_loss(x_pred, x1, top_k=top_k)
                pair_loss = pair_loss + self.lambda_density * den_v
                den_sum = den_sum + den_v.detach()

            if self.lambda_energy > 0:
                eng_v = energy(self._step_norms(x0, t0, t1))
                pair_loss = pair_loss + self.lambda_energy * eng_v
                eng_sum = eng_sum + eng_v.detach()

            total = total + pair_loss

        self.log(f"{phase}_loss", total, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_ot_loss", ot_sum, on_step=False, on_epoch=True)
        self.log(f"{phase}_density_loss", den_sum, on_step=False, on_epoch=True)
        self.log(f"{phase}_energy_loss", eng_sum, on_step=False, on_epoch=True)
        return {"loss": total}

    def training_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, phase="test")
        self.log("test_loss", out["loss"], prog_bar=True, on_epoch=True)
        return out

    # ------------------------------------------------------------------ #
    # Inference surface
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """Decoded prediction after integrating over ``integration_times``."""
        assert self.network is not None, "Network not configured. Call setup() first."
        t_span = torch.tensor(self.integration_times, device=x.device, dtype=x.dtype)
        x_hat, _z_T = self.network(x, t_span)
        return x_hat

    def encode(self, x: Tensor) -> Tensor:
        """Latent embedding ``z_T`` — what experiment.py reads into embeddings.

        Returns shape ``(n, latent_dim)``.
        """
        assert self.network is not None, "Network not configured. Call setup() first."
        t_span = torch.tensor(self.integration_times, device=x.device, dtype=x.dtype)
        return self.network.encode(x, t_span)

    def integrate(self, x0: Tensor, t_start: float, t_end: float) -> Tensor:
        """Flow the data cloud ``x0`` from ``t_start`` to ``t_end`` (data space).

        Runs encode -> integrate latent over ``[t_start, t_end]`` -> decode and
        returns the decoded predicted cloud. Used for drift-recovery checks.
        """
        assert self.network is not None, "Network not configured. Call setup() first."
        t_span = torch.tensor([t_start, t_end], device=x0.device, dtype=x0.dtype)
        x_pred, _z_T = self.network(x0, t_span)
        return x_pred

    def extra_outputs(self) -> dict:
        """DEFERRED: trajectory / GRN extras not built in v0."""
        return {}

    # ------------------------------------------------------------------ #
    # Optimizer (mirrors LatentODE)
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        if isinstance(self._optimizer_partial, functools.partial):
            optimizer = self._optimizer_partial(self.parameters())
        else:
            optimizer_partial = hydra_zen.instantiate(self._optimizer_partial)
            optimizer = optimizer_partial(self.parameters())

        max_epochs = getattr(self.trainer, "max_epochs", None)
        if not max_epochs or max_epochs < 1:
            max_epochs = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
