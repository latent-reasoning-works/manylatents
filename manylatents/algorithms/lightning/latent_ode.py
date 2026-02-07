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
        network: Hydra config or instantiated LatentODENetwork.
        optimizer: Hydra config for optimizer (partial instantiation).
        loss: Hydra config or instantiated loss module.
        datamodule: Data module for loading train/val/test data.
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

        self.save_hyperparameters(ignore=["datamodule", "network", "loss"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        """Infer input_dim from data if needed, then build network."""
        if self.network is not None:
            return  # Already configured (idempotent)
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None and self.datamodule is not None:
                first_batch = next(iter(self.datamodule.train_dataloader()))
                data = first_batch["data"] if isinstance(first_batch, dict) else first_batch[0]
                self.network_config["input_dim"] = data.shape[1]
        self.configure_model()

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

        if hasattr(self.loss_fn, "components"):
            extras = {"latent": z_T, "raw": x}
            comps = self.loss_fn.components(outputs=x_hat, targets=x, **extras)
            loss = sum(comps.values())
            self.log_dict(
                {f"{phase}_{k}": v for k, v in comps.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
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
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
