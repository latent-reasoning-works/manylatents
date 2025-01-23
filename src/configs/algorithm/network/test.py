"""An example of a simple fully connected network."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class _testnet:
    """
    Dataclass containing the network hyperparameters.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        use_bias (bool): Whether to include a bias term in the linear layer.
        activation (str): Activation function to use (e.g., 'ReLU', 'Sigmoid').
    """
    input_dim: int = 10
    output_dim: int = 2
    use_bias: bool = True
    activation: str = "ReLU"


class testnet(nn.Module):
    """
    A simple one-layer fully connected neural network compatible with PyTorch and PyTorch Lightning.

    Architecture:
        - Linear Layer
        - Activation Function

    Args:
        hparams (SimpleFcNetHParams): Hyperparameters for the network.
    """

    def __init__(
        self,
        hparams: Optional[_testnet] = None,
    ):
        super(testnet, self).__init__()
        self.hparams = hparams or _testnet()

        # Define the linear layer
        self.layer = nn.Linear(
            self.hparams.input_dim,
            self.hparams.output_dim,
            bias=self.hparams.use_bias
        )

        # Define the activation function
        if hasattr(nn, self.hparams.activation):
            self.activation = getattr(nn, self.hparams.activation)()
        else:
            raise ValueError(f"Unknown activation function: {self.hparams.activation}")

        # Define loss function (optional, can be handled in Trainer)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        x = self.layer(x)
        x = self.activation(x)
        return x