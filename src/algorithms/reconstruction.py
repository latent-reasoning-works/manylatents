import functools
from collections.abc import Sequence
from logging import getLogger

import hydra_zen
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger = getLogger(__name__)


class Reconstruction(LightningModule):
    """
    An algorithm for reconstruction tasks that wraps a neural network
    (e.g. one of the AAnet variants) which is specified by a Hydra config.    
    """

    def __init__(
        self,
        datamodule,
        network: hydra_zen.Config,  # Hydra config for a torch.nn.Module (e.g. an AAnet variant)
        optimizer: hydra_zen.Config,  # Hydra config for the optimizer (a functools.partial)
        init_seed: int = 42,
    ):
        """
        Parameters:
            datamodule: Object used to load train/val/test data.
            network: The config of the network (e.g. AAnet_VAE or AAnet_vanilla) to instantiate.
            optimizer: The config for the optimizer.
            init_seed: Seed for deterministic weight initialization.
        """
        super().__init__()
        self.datamodule = datamodule
        self.network_config = network
        self.optimizer_config = optimizer
        self.init_seed = init_seed

        # Save hyper-parameters (except datamodule, which may be non-serializable).
        self.save_hyperparameters(ignore=["datamodule"])
        
        # This will hold our instantiated network.
        self.network: torch.nn.Module | None = None

    def configure_model(self):
        """
        Instantiate the network from the Hydra config and initialize weights.
        This should be called (or implicitly invoked) before training starts.
        """
        # Example input used for shape inference.
        self.example_input_array = torch.zeros((self.datamodule.batch_size, *self.datamodule.dims))
        with torch.random.fork_rng():
            # Deterministic weight initialization.
            torch.manual_seed(self.init_seed)
            self.network = hydra_zen.instantiate(self.network_config)
            # If the network has lazy weights, do a forward pass to initialize them.
            if any(torch.nn.parameter.is_lazy(p) for p in self.network.parameters()):
                _ = self.network(self.example_input_array)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass simply delegates to the underlying network.
        """
        assert self.network is not None, "Network not configured. Call configure_model() first."
        return self.network(x)

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """
        Run a training step: forward the input through the network and compute its loss.
        """
        # Assume the first element of the batch is the input.
        x = batch[0]
        outputs = self.network(x)
        # Delegate loss calculation to the network (it should implement loss_function)
        loss = self.network.loss_function(outputs, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        """
        Run a validation step: forward the input and compute its loss.
        """
        x = batch[0]
        outputs = self.network(x)
        loss = self.network.loss_function(outputs, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Instantiate the optimizer using the provided Hydra config.
        """
        optimizer_partial: functools.partial = hydra_zen.instantiate(self.optimizer_config)
        optimizer: Optimizer = optimizer_partial(self.parameters())
        return optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """
        Optionally, return callbacks to be used during training.
        """
        # For example, you could attach custom callbacks here.
        return []
