"""Trajectory losses ported from the MIOFlow family used by Cflows.

Reference: ``KrishnaswamyLab/cflows`` file ``BEMIOflow/MIOFlow/losses.py``
(``OT_loss``, ``MMD_loss``, ``Density_loss``). These are the correctness
anchors of the trajectory-inference pipeline, so the numerics here match the
reference exactly; deviations are documented inline.

Each class mirrors the ``nn.Module`` + ``forward(...)`` shape of
``losses/mse.py`` so it is ``_target_``-instantiable like ``MSELoss``.
"""

from __future__ import annotations

import ot
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["OTLoss", "MMDLoss", "DensityLoss", "energy"]


class OTLoss(nn.Module):
    """Optimal-transport loss ``<pi, M>`` between two point clouds.

    Cost ``M_ij = ||A_i - B_j||^2`` (``torch.cdist(A, B) ** 2``). The transport
    plan ``pi = solver(a, b, M)`` is **detached** (no gradient flows through the
    plan; gradients flow only through ``M``). The loss is ``(pi * M).sum()``.

    Supported solvers:
      * ``"emd"``        -> ``ot.emd``
      * ``"sinkhorn"``   -> ``ot.sinkhorn`` with ``reg = sinkhorn_lambda`` (2.0)
      * ``"unbalanced"`` -> ``ot.unbalanced.sinkhorn_knopp_unbalanced`` (1.0, 1.0)

    Marginals default to uniform ``a = b = 1/n``; ``source_mass`` /
    ``target_mass`` override them (each is renormalised to sum to 1).
    """

    # Deviation from reference: the reference exposes the raw POT function name
    # ``sinkhorn_knopp_unbalanced``. The task spec asks for the alias
    # ``unbalanced``; we accept both so callers of either vintage work.
    _SOLVERS = ("emd", "sinkhorn", "unbalanced", "sinkhorn_knopp_unbalanced")

    def __init__(self, which: str = "emd", sinkhorn_lambda: float = 2.0):
        super().__init__()
        if which not in self._SOLVERS:
            raise ValueError(f"{which!r} not known (valid: {self._SOLVERS})")
        self.which = which
        self.sinkhorn_lambda = sinkhorn_lambda

    def _solve(self, a: Tensor, b: Tensor, M: Tensor) -> Tensor:
        if self.which == "emd":
            return ot.emd(a, b, M)
        if self.which == "sinkhorn":
            return ot.sinkhorn(a, b, M, self.sinkhorn_lambda)
        # "unbalanced" / "sinkhorn_knopp_unbalanced"
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, 1.0, 1.0)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mass: Tensor | None = None,
        target_mass: Tensor | None = None,
        return_plan: bool = False,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        device = source.device

        if source_mass is None:
            a = torch.full(
                (source.shape[0],),
                1.0 / source.shape[0],
                dtype=source.dtype,
                device=device,
            )
        else:
            a = source_mass / source_mass.sum()

        if target_mass is None:
            b = torch.full(
                (target.shape[0],),
                1.0 / target.shape[0],
                dtype=target.dtype,
                device=device,
            )
        else:
            b = target_mass / target_mass.sum()

        M = torch.cdist(source, target) ** 2

        # Solve on CPU with detached inputs; the plan carries no gradient.
        # Deviation from reference: the reference used a ``use_cuda`` flag to
        # ``.cuda()`` the plan. We instead infer the device from the inputs so
        # the loss runs on CPU / CUDA / MPS with no flag.
        pi = self._solve(a.detach().cpu(), b.detach().cpu(), M.detach().cpu())
        if not isinstance(pi, Tensor):
            pi = torch.as_tensor(pi, dtype=M.dtype)
        pi = pi.clone().detach().to(device=device, dtype=M.dtype)

        loss = torch.sum(pi * M)
        if return_plan:
            return loss, pi
        return loss


class MMDLoss(nn.Module):
    """Multi-scale Gaussian-kernel Maximum Mean Discrepancy (MMD^2).

    Ported verbatim from the reference (originally from
    ``ZongxianLee/MMD_Loss.Pytorch``). Uses ``kernel_num`` (5) Gaussian kernels
    whose bandwidths are geometric multiples of ``kernel_mul`` (2.0) around a
    median-style bandwidth estimated from the pooled pairwise L2 distances
    (or ``fix_sigma`` when set).
    """

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(
        self,
        source: Tensor,
        target: Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
    ) -> Tensor:
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth = bandwidth / kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source: Tensor, target: Tensor, **kwargs: object) -> Tensor:
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class DensityLoss(nn.Module):
    """Hinged top-k nearest-neighbour density loss.

    ``relu(topk_smallest(cdist(pred, target), k=top_k) - hinge).mean()``.

    For each predicted point, the ``top_k`` smallest distances to the target
    cloud are taken; distances within ``hinge_value`` incur no penalty, and the
    excess is averaged. Zero when every prediction is within ``hinge_value`` of
    at least ``top_k`` targets.

    ``groups`` / ``to_ignore`` reproduce the reference's *global* variant: when
    ``groups`` is given, ``source`` and ``target`` are treated as per-timepoint
    sequences and the loss is stacked over groups ``1..len(groups)`` excluding
    ``to_ignore``. Left ``None`` (the default) it is the *local* loss above.
    """

    def __init__(self, hinge_value: float = 0.01):
        super().__init__()
        self.hinge_value = hinge_value

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        groups: object | None = None,
        to_ignore: object | None = None,
        top_k: int = 5,
        **kwargs: object,
    ) -> Tensor:
        if groups is not None:
            # global loss: source/target are per-timepoint sequences
            c_dist = torch.stack(
                [
                    torch.cdist(source[i], target[i])
                    for i in range(1, len(groups))
                    if groups[i] != to_ignore
                ]
            )
        else:
            # local loss
            c_dist = torch.stack([torch.cdist(source, target)])

        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        # Deviation from reference: the reference clamps negatives in place
        # (``values[values < 0] = 0``), which breaks autograd. ``clamp(min=0)``
        # is the same relu numerically and keeps gradients intact.
        values = torch.clamp(values - self.hinge_value, min=0.0)
        loss = torch.mean(values)
        return loss


def energy(norms: list[Tensor]) -> Tensor:
    """Trajectory energy penalty ``Sum_t ||f_theta(t, x_t)||^2``.

    This is *not* a standalone loss: during ODE integration the model appends
    the per-step squared velocity norm ``||f||^2`` to a list, and this helper
    sums those per-step contributions into a single scalar. Each element of
    ``norms`` is an already-squared norm (a scalar or a per-sample tensor); the
    result is the total summed over all steps (and samples).
    """
    if len(norms) == 0:
        return torch.zeros(())
    return torch.stack([n.sum() for n in norms]).sum()
