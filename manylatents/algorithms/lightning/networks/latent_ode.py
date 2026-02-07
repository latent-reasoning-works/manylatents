"""Latent ODE network: encode → ODE integrate → decode.

Network components for the Latent ODE architecture. This is the nn.Module
(the network), not the training wrapper. Analogous to how Autoencoder is
the network used by Reconstruction.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ODEFunc(nn.Module):
    """Learned vector field dz/dt = f(t, z).

    A neural network that outputs the time derivative of the latent state.
    The forward signature is f(t, z) as required by torchdiffeq.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = latent_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.Tanh()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.net(z)


class LatentODENetwork(nn.Module):
    """Complete Latent ODE: encode → integrate dz/dt = f(t,z) → decode.

    Analogous to Autoencoder but with an ODE solver between encoder and decoder.

    Args:
        input_dim: Input feature dimension.
        latent_dim: Latent state dimension (ODE state size).
        hidden_dim: Hidden width in the ODE vector field.
        encoder_hidden_dims: Hidden layer sizes for encoder MLP.
        decoder_hidden_dims: Hidden layer sizes for decoder MLP.
        ode_n_layers: Number of hidden layers in ODEFunc.
        solver: ODE solver name ('dopri5', 'euler', 'rk4', etc.).
        rtol: Relative tolerance for adaptive solvers.
        atol: Absolute tolerance for adaptive solvers.
        use_adjoint: If True, use odeint_adjoint for O(1) memory backprop.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        encoder_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        ode_n_layers: int = 2,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_adjoint: bool = True,
    ):
        super().__init__()
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256]

        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        # Encoder: input_dim -> latent_dim
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ODE vector field
        self.ode_func = ODEFunc(latent_dim, hidden_dim, ode_n_layers)

        # Decoder: latent_dim -> input_dim
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in decoder_hidden_dims:
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def _integrate(self, z_0: Tensor, t_span: Tensor) -> Tensor:
        """Run ODE solver. Returns trajectory (n_times, batch, latent_dim)."""
        if self.use_adjoint:
            from torchdiffeq import odeint_adjoint as odeint_fn
        else:
            from torchdiffeq import odeint as odeint_fn
        return odeint_fn(
            self.ode_func,
            z_0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )

    def forward(
        self, x: Tensor, t_span: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """Full forward: encode → integrate → decode.

        Args:
            x: (batch, input_dim) input data.
            t_span: Integration time points. Default [0, 1].

        Returns:
            x_hat: (batch, input_dim) reconstruction from z_T.
            z_T: (batch, latent_dim) ODE endpoint embedding.
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device)
        z_0 = self.encoder(x)
        z_traj = self._integrate(z_0, t_span)
        z_T = z_traj[-1]
        x_hat = self.decoder(z_T)
        return x_hat, z_T

    def encode(self, x: Tensor, t_span: Optional[Tensor] = None) -> Tensor:
        """Encode → integrate → return z_T. No decoder.

        This is what experiment.py calls for embedding extraction.

        Returns:
            z_T: (batch, latent_dim) ODE endpoint.
        """
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device)
        z_0 = self.encoder(x)
        z_traj = self._integrate(z_0, t_span)
        return z_traj[-1]

    def get_latent_trajectory(self, x: Tensor, t_eval: Tensor) -> Tensor:
        """Full trajectory for Geomancy per-timepoint geometric profiling.

        Args:
            x: (batch, input_dim) input data.
            t_eval: (n_times,) evaluation timepoints.

        Returns:
            z_traj: (n_times, batch, latent_dim) trajectory.
        """
        z_0 = self.encoder(x)
        return self._integrate(z_0, t_eval)
