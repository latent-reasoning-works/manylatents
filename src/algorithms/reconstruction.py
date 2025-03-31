import functools
from collections.abc import Sequence
from logging import getLogger
from typing import Sequence, Union

import hydra_zen
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

logger = getLogger(__name__)


class Reconstruction(LightningModule):
    """
    An algorithm for reconstruction tasks that wraps a neural network
    (e.g. one of the AAnet variants) specified by a Hydra config.
    """

    def __init__(
        self,
        datamodule,
        network: DictConfig,
        optimizer: DictConfig,
        init_seed: int = 42,
    ):
        """
        Parameters:
            datamodule: Object used to load train/val/test data.
            network: The config of the network (e.g. AAnet or Autoencoder) to instantiate.
            optimizer: The config for the optimizer.
            init_seed: Seed for deterministic weight initialization.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        self.save_hyperparameters(ignore=["datamodule"])
        self.network: torch.nn.Module | None = None

    def configure_model(self):
        """
        Instantiate the network from the Hydra config and initialize weights.
        Checks whether the network configuration is still a config or already instantiated.
        """
        with torch.random.fork_rng():
            torch.manual_seed(self.init_seed)
            if isinstance(self.network_config, (dict, DictConfig)) or OmegaConf.is_config(self.network_config):
                self.network = hydra_zen.instantiate(self.network_config)
            else:
                self.network = self.network_config

            # If the network has lazy layers, perform a forward pass with a dummy input.
            if any(torch.nn.parameter.is_lazy(p) for p in self.network.parameters()):
                if hasattr(self.datamodule, "batch_size"):
                    dummy_shape = getattr(self.datamodule, "dims", None)
                    if dummy_shape is None and hasattr(self, "dummy_input_shape"):
                        dummy_shape = self.dummy_input_shape
                    if dummy_shape is not None:
                        example_input = torch.zeros((self.datamodule.batch_size, *dummy_shape))
                        _ = self.network(example_input)
                    else:
                        logger.info("No dummy input shape provided; skipping lazy weight initialization.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Delegate forward pass to the underlying network.
        """
        assert self.network is not None, "Network not configured. Call configure_model() first."
        return self.network(x)

    def shared_step(self, batch: tuple[Tensor, ...], batch_idx: int, phase: str) -> dict:
        """
        Common step logic for training, validation, and testing.
        """
        x = batch[0]
        outputs = self.network(x)
        loss = self.network.loss_function(outputs)
        self.log(f"{phase}_loss", loss, prog_bar=True)
        return {"loss": loss, "outputs": outputs}

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> dict:
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

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """
        Optionally, return callbacks to be used during training.
        """
        return []
