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
                 network: DictConfig, 
                 loss: DictConfig,
                 optimizer: DictConfig, 
                 init_seed: int = 42):
        """
        Parameters:
            datamodule: Object used to load train/val/test data.
            network: The config of the network (e.g. AAnet or Autoencoder) to instantiate. Must include 'input_dim'.
            optimizer: The config for the optimizer.
            init_seed: Seed for deterministic weight initialization.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed
        self.loss_config = loss

        self.save_hyperparameters(ignore=["datamodule"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        """
        Set up the network using the provided network config.
        """
        # 1) Infer feature-dim if not provided, and write it back into the config
        if self.network_config.input_dim is None:
            first_batch = next(iter(self.datamodule.train_dataloader()))["data"]
            feat_dim = first_batch.shape[1]
            # Patch the DictConfig so that instantiate() sees a real int
            self.network_config.input_dim = feat_dim
        else:
            feat_dim = self.network_config.input_dim

        # 2) Now instantiate with a concrete input_dim
        self.configure_model()

        logger.info(
            f"Reconstruction network configured with input_dim={feat_dim}"
        )
        

    def configure_model(self):
        """
        Instantiate the network from the Hydra config.
        Assumes that 'input_dim' is already set in the config.
        """
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
