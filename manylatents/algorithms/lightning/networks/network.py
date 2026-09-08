from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class HasForward(Protocol):
    def forward(self, x: torch.Tensor, **kwargs) -> Any: ...


@runtime_checkable
class HasEncode(Protocol):
    """Fixed-coordinate encoder contract (implementations must also be nn.Module).

    ``encode`` maps (batch, ambient_dim) to (batch, latent_dim). Decoding is
    optional: latent flow integration never needs it. Consumers may use an
    optional ``decode(z)`` for ambient-space trajectory output. Check
    ``isinstance(encoder, HasDecode)`` to discover this structural capability.
    """

    latent_dim: int

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class HasDecode(Protocol):
    """Optional decoder: (batch, latent_dim) -> (batch, ambient_dim).

    This structural check does not execute the decoder or validate its output.
    """

    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
