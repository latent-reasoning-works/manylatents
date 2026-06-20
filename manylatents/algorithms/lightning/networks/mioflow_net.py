"""MIOFlow network components: ODEFunc and loss functions.

MIOFlow (Manifold Interpolating Optimal-Transport Flows) learns a Neural ODE
velocity field over time-labeled cell populations. The ODEFunc takes [t, x] as
input (time concatenated), unlike the existing LatentODE ODEFunc which ignores t.

Reference: Huguet et al., arXiv:2206.14928 (2022)
"""

import torch
import torch.nn as nn
from torch import Tensor


class MIOFlowODEFunc(nn.Module):
    """Learned velocity field dx/dt = f(t, x) with time conditioning.

    Takes [t, x] concatenated as input, producing a time-dependent flow.

    Args:
        input_dim: Spatial dimensionality of the data.
        hidden_dim: Width of hidden layers in the MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t_expanded = t.expand(x.size(0), 1)
        return self.net(torch.cat([t_expanded, x], dim=-1))


def mioflow_ot_loss(source: Tensor, target: Tensor) -> Tensor:
    """Earth Mover Distance (2-Wasserstein) between two point clouds.

    Uses POT library for exact OT computation.
    """
    import ot

    mu = torch.tensor(ot.unif(source.size(0)), dtype=source.dtype, device=source.device)
    nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
    M = torch.cdist(source, target) ** 2
    return ot.emd2(mu, nu, M)


def mioflow_energy_loss(model: MIOFlowODEFunc, x0: Tensor, t_seq: Tensor) -> Tensor:
    """Penalizes large velocity magnitudes along the ODE trajectory."""
    from torchdiffeq import odeint

    trajectory = odeint(model, x0, t_seq)
    total_energy = 0.0
    num_evaluations = 0
    for i, t_val in enumerate(t_seq):
        x_t = trajectory[i]
        dx_dt = model(t_val, x_t)
        total_energy = total_energy + torch.sum(dx_dt**2)
        num_evaluations += x_t.size(0)
    return total_energy / num_evaluations


def mioflow_density_loss(
    source: Tensor, target: Tensor, top_k: int = 5, hinge_value: float = 0.01
) -> Tensor:
    """k-NN hinge loss enforcing density matching between distributions."""
    c_dist = torch.cdist(source, target)
    values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
    return torch.mean(torch.clamp(values - hinge_value, min=0.0))
