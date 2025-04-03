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
    'input_shape' provided via the config.
    """
    def __init__(self, datamodule, network: DictConfig, optimizer: DictConfig, init_seed: int = 42):
        """
        Parameters:
            datamodule: Object used to load train/val/test data.
            network: The config of the network (e.g. AAnet or Autoencoder) to instantiate. Must include 'input_shape'.
            optimizer: The config for the optimizer.
            init_seed: Seed for deterministic weight initialization.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        self.save_hyperparameters(ignore=["datamodule"])
        self.network: nn.Module | None = None

    def setup(self, stage=None):
        """
        Set up the network using the provided network config.
        """
        self.configure_model()
        logger.info(
            f"Reconstruction network configured with input shape: {self.network_config.input_shape}"
        )

    def configure_model(self):
        """
        Instantiate the network from the Hydra config.
        Assumes that 'input_shape' is already set in the config.
        """
        torch.manual_seed(self.init_seed)
        if isinstance(self.network_config, (dict, DictConfig)):
            self.network = hydra_zen.instantiate(self.network_config)
        else:
            self.network = self.network_config

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
        if hasattr(self.network, "encoder") and callable(self.network.encoder):
            return self.network.encoder(x)
        elif hasattr(self.network, "encode") and callable(self.network.encode):
            return self.network.encode(x)
        else:
            raise NotImplementedError("The underlying network does not support encoding.")

    def shared_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int, phase: str) -> dict:
        """
        Common step logic for training, validation, and testing.
        """
        x = batch["data"]
        outputs = self.network(x)
        loss = self.network.loss_function(outputs)
        self.log(f"{phase}_loss", loss, prog_bar=True)
        return {"loss": loss, "outputs": outputs}

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="test")

    def configure_optimizers(self):
        """
        Instantiate the optimizer using the provided Hydra config.
        """
        if isinstance(self.optimizer_config, functools.partial):
            optimizer_partial = self.optimizer_config
        else:
            optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        return optimizer_partial(self.parameters())
