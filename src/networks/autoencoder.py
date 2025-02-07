import torch
import numpy as np
from lightning import LightningModule
from src.networks.BaseAE import BaseAE


class AE(BaseAE):
    def __init__(
        self,
        dim,
        emb_dim,
        width=64,
        activation_fn=torch.nn.ReLU(),
    ):
        super().__init__(dim, emb_dim, width=width, activation_fn=activation_fn)

    def forward(self, x):
        z = self.encoder(x)
        return [self.decode(z), z]

    def loss_function(self, inputs, outputs):
        """output are the outputs of forward method (decoder)"""
        loss = torch.tensor(0.0)
        x_hat, z = outputs

        loss = torch.nn.functional.mse_loss(inputs, x_hat)
        logs = {"loss": loss}
        logs = {k: v.item() for k, v in logs.items()}
        return loss, logs

    @torch.no_grad()
    def generate(self, x):
        return self.decode(self.encode(x))
    

class LitAE(LightningModule):
    def __init__(self, model, optimizer, lr=1e-3, noise=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.save_hyperparameters()
        self.noise = noise

    def configure_optimizers(self):
        return self.optimizer

    def step(self, data, noise=None):
        if noise is not None: 
            model_input = data + torch.randn_like(data) * noise
        else:
            model_input = data

        model_output = self.model(model_input)
        loss, logs = self.model.loss_function(data, model_output)

        return loss, logs

    def training_step(self, batch):
        batch_size = batch.shape[0]
        loss, logs = self.step(batch, noise=self.noise)
        self.log("train_loss", loss.item(), logger=True, batch_size=batch_size)
        self.log_dict(logs, logger=True, batch_size=batch_size)
        return {"loss": loss}

    def test_val_step(self, batch, prefix):
        batch_size = batch.shape[0]
        loss, logs = self.step(batch, noise=None)
        self.log(
            f"{prefix}_loss",
            loss.item(),
            logger=True,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(
            {f"{prefix}_{k}": v for k, v in logs.items()},
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss}

    def validation_step(self, batch):
        return self.test_val_step(batch, "val")

    def test_step(self, batch):
        return self.test_val_step(batch, "test")