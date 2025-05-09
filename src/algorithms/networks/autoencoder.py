from typing import List, Union

import torch.nn as nn


class Autoencoder(nn.Module):
    """
    A simple autoencoder for reconstruction tasks.
    """

    def __init__(self, input_dim: int, 
                 hidden_dims: Union[List[int], int], 
                 latent_dim: int, 
                 activation: str = "relu"):
        """
        Parameters:
            input_dim (int): Size of the input layer.
            hidden_dims (list[int]): List specifying the number of units in each encoder hidden layer.
            latent_dim (int): Size of the latent bottleneck representation.
            activation (str): Activation function ("relu", "tanh", or "sigmoid").
        """
        super().__init__()
        
        ## allow single, int hidden dim
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Activation function mapping
        activation_fn = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }.get(activation.lower(), nn.ReLU())

        # Encoder: progressively reducing dimensions
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(activation_fn)
            prev_dim = h_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))  # Bottleneck layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: progressively increasing dimensions
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(activation_fn)
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # Output layer (same as input size)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def loss_function(self, outputs, targets):
        """Simple MSE loss function for reconstruction."""
        return nn.MSELoss()(outputs, targets)
