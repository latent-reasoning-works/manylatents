from typing import List, Union

import torch.nn as nn
from torch import Tensor


class Autoencoder(nn.Module):
    """
    A simple autoencoder for reconstruction tasks.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: Union[List[int], int], 
        latent_dim: int, 
        activation: str = "relu",
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        """
        Parameters:
            input_dim (int): Size of the input layer.
            hidden_dims (list[int] or int):  Number of units in each encoder hidden layer.
            latent_dim (int): Size of the latent bottleneck representation.
            activation (str): "relu", "tanh", or "sigmoid".
            batchnorm (bool): If True, insert BatchNorm1d after each Linear.
            dropout (float): Dropout probability after each activation (0=no dropout).
        """
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.input_dim   = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim  = latent_dim

        # pick activation
        act = {
            "relu":    nn.ReLU(),
            "tanh":    nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }.get(activation.lower(), nn.ReLU())

        # build encoder
        encoder_layers = []
        prev = input_dim
        for h in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev, h))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(act)
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # build decoder
        decoder_layers = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev, h))
            if batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(act)
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor, return_latent: bool = False) -> Tensor:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return (x_hat, z) if return_latent else x_hat

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)
