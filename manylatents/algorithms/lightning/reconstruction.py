import functools
import logging

import hydra_zen
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class Reconstruction(LightningModule):
    """
    An algorithm for reconstruction tasks that wraps a neural network (e.g. AAnet variants or Autoencoder)
    specified by a Hydra config. This version assumes that the network configuration includes an
    'input_dim' provided via the config.
    """
    def __init__(self, datamodule, 
                 network: DictConfig | dict | nn.Module,
                 loss: DictConfig,
                 optimizer: DictConfig, 
                 init_seed: int = 42):
        """
        Parameters:
            datamodule: Object used to load train/val/test data.
            network: Network config (e.g. AAnet or Autoencoder), or an existing
                module whose identity and weights are preserved. Seed an existing
                module at construction, e.g. Autoencoder(..., init_seed=42).
            optimizer: The config for the optimizer.
            init_seed: Seed before config construction; never resets supplied weights.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed
        self.loss_config = loss

        self.save_hyperparameters(ignore=["datamodule", "network", "loss"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        """
        Set up the network using the provided network config.
        """
        if self.network is not None:
            return
        if isinstance(self.network_config, (dict, DictConfig)):
            if self.network_config.get("input_dim") is None and self.datamodule is not None:
                first_batch = next(iter(self.datamodule.train_dataloader()))["data"]
                self.network_config["input_dim"] = first_batch.shape[1]
        self.configure_model()
        

    def configure_model(self):
        """
        Instantiate the network from the Hydra config.
        Assumes that 'input_dim' is already set in the config.
        """
        # Lightning calls this hook for every stage, even when setup() returns early.
        if self.network is not None:
            return
        torch.manual_seed(self.init_seed)

        cfg_map = {
            "network": self.network_config,
            "loss_fn": self.loss_config,
            # add more as needed 
        }

        for attr, cfg in cfg_map.items():
            if isinstance(cfg, (dict, DictConfig)):
                inst = hydra_zen.instantiate(cfg)
            else:
                inst = cfg  # already an object
            setattr(self, attr, inst)

        # stash the optimizer config for the actual optimizer_step
        # leave the torch.optim instantiation for configure_optimizers        
        self._optimizer_partial = self.optimizer_config

        logger.info(f"Instantiated network={self.network.__class__.__name__}, "
                    f"loss_fn={self.loss_fn.__class__.__name__}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Delegate the forward pass to the underlying network.
        """
        assert self.network is not None, "Network not configured. Call configure_model() first."
        return self.network(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent representation produced by the network's encoder.
        """
        assert self.network is not None, "Network not configured. Call configure_model() first."
        return self.network.encode(x)

    def shared_step(self, batch, batch_idx, phase: str) -> dict:
        x       = batch["data"]
        outputs = self.network(x)
        extras  = {"latent": self.network.encode(x), "raw": x}

        # if our loss has .components(), pull them out and log
        if hasattr(self.loss_fn, "components"):
            comps = self.loss_fn.components(outputs=outputs, targets=x, **extras)
            total = sum(comps.values())
            # log each piece e.g. train_recon, train_pr, etc.
            self.log_dict(
                {f"{phase}_{k}": v for k, v in comps.items()},
                on_step=False, on_epoch=True, prog_bar=False
            )
            loss = total
        else:
            loss = self.loss_fn(outputs=outputs, targets=x, **extras)

        # still log the aggregate
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "outputs": outputs}

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, batch_idx, phase="test")
        # out["loss"] is your test loss
        self.log("test_loss", out["loss"], prog_bar=True, on_epoch=True)
        return out

    def configure_optimizers(self):
        """
        Instantiate the optimizer using the provided Hydra config.
        """
        if isinstance(self.optimizer_config, functools.partial):
            optimizer_partial = self.optimizer_config
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        return optimizer_partial(self.parameters())
