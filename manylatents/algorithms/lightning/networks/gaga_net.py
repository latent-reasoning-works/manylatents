"""GAGA network components: Preprocessor, Autoencoder, and loss functions.

GAGA (Geometry-Aware Generative Autoencoder) learns low-dimensional embeddings
that preserve pairwise distances and/or affinity structure from high-dimensional
data. The architecture supports three loss modes: distance-preserving (AEDist),
affinity-preserving (AEProb), and combined.

Reference: Huguet et al., "Geodesic Sinkhorn for Fast and Accurate
Optimal Transport on Manifolds" (2023)
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Preprocessor(nn.Module):
    """Stores dataset statistics for input/distance normalization.

    Buffers are persistent so they travel with ``state_dict`` and device moves.

    Args:
        mean: Per-feature mean of the training data.
        std: Per-feature standard deviation of the training data.
        dist_std: Standard deviation of pairwise distances in the training data.
    """

    def __init__(
        self,
        mean: float | Tensor = 0.0,
        std: float | Tensor = 1.0,
        dist_std: float | Tensor = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.as_tensor(mean, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "std", torch.as_tensor(std, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "dist_std", torch.as_tensor(dist_std, dtype=torch.float32), persistent=True
        )

    def normalize(self, x: Tensor) -> Tensor:
        """Center and scale features to zero-mean / unit-variance."""
        return (x - self.mean) / self.std

    def unnormalize(self, x: Tensor) -> Tensor:
        """Invert ``normalize``."""
        return x * self.std + self.mean

    def normalize_dist(self, d: Tensor) -> Tensor:
        """Scale distances by the training distance standard deviation."""
        return d / self.dist_std


class GAGAAutoencoder(nn.Module):
    """MLP autoencoder with optional spectral normalization.

    Mirrors the ``Autoencoder`` in ``autoencoder.py`` but always returns
    ``(x_hat, z)`` from ``forward`` and supports spectral normalization on
    all ``Linear`` layers.

    Args:
        input_dim: Size of the input layer.
        latent_dim: Size of the latent bottleneck representation.
        hidden_dims: Number of units in each encoder hidden layer.
            The decoder uses the reversed order.
        activation: ``"relu"``, ``"tanh"``, or ``"sigmoid"``.
        batchnorm: If ``True``, insert ``BatchNorm1d`` after each ``Linear``.
        dropout: Dropout probability after each activation (0 = no dropout).
        spectral_norm: If ``True``, wrap every ``Linear`` with spectral
            normalization for Lipschitz-constrained training.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Union[List[int], int] = 128,
        activation: str = "relu",
        batchnorm: bool = False,
        dropout: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        def _linear(in_f: int, out_f: int) -> nn.Module:
            layer = nn.Linear(in_f, out_f)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            return layer

        # --- encoder ---
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            encoder_layers.append(_linear(prev, h))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(act_cls())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev = h
        encoder_layers.append(_linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- decoder (mirrored architecture) ---
        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.append(_linear(prev, h))
            if batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(act_cls())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev = h
        decoder_layers.append(_linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        """Map input to latent space."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Map latent code back to input space."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode then decode, returning ``(x_hat, z)``."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def gaga_distance_loss(
    z: Tensor,
    gt_distances_upper: Tensor,
    dist_mse_decay: float = 0.0,
) -> Tensor:
    """MSE between latent pairwise distances and ground-truth distances.

    Args:
        z: ``(B, d)`` latent embeddings.
        gt_distances_upper: ``(B*(B-1)/2,)`` ground-truth pairwise distances
            (upper-triangular, as returned by ``torch.nn.functional.pdist``).
        dist_mse_decay: Exponential decay factor that down-weights distant
            pairs.  ``0`` means uniform weighting.
    """
    dist_emb = F.pdist(z)
    if dist_mse_decay > 0:
        weights = torch.exp(-dist_mse_decay * gt_distances_upper)
        return ((dist_emb - gt_distances_upper) ** 2 * weights).mean()
    return F.mse_loss(dist_emb, gt_distances_upper)


def gaga_reconstruction_loss(x_hat: Tensor, x: Tensor) -> Tensor:
    """MSE reconstruction loss (thin wrapper for naming consistency)."""
    return F.mse_loss(x_hat, x)


def gaga_affinity_loss(
    pred_prob: Tensor,
    gt_prob: Tensor,
    loss_type: str = "kl",
    eps: float = 1e-8,
) -> Tensor:
    """Divergence between predicted and ground-truth probability matrices.

    Args:
        pred_prob: ``(N, N)`` predicted row-stochastic matrix.
        gt_prob: ``(N, N)`` ground-truth row-stochastic matrix.
        loss_type: ``"kl"`` (KL divergence), ``"jsd"`` (Jensen-Shannon), or
            ``"mse"`` (mean squared error).
        eps: Small constant for numerical stability in log computations.
    """
    if loss_type == "kl":
        log_pred = torch.log(pred_prob + eps)
        return F.kl_div(log_pred, gt_prob + eps, reduction="batchmean", log_target=False)
    elif loss_type == "jsd":
        m = 0.5 * (pred_prob + gt_prob)
        kl1 = F.kl_div(torch.log(pred_prob + eps), m, reduction="batchmean", log_target=False)
        kl2 = F.kl_div(torch.log(gt_prob + eps), m, reduction="batchmean", log_target=False)
        return 0.5 * (kl1 + kl2)
    elif loss_type == "mse":
        return F.mse_loss(pred_prob, gt_prob)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Use 'kl', 'jsd', or 'mse'.")


def compute_prob_matrix(
    z: Tensor,
    kernel_method: str = "gaussian",
    alpha: float = 1.0,
    bandwidth: float = 1.0,
) -> Tensor:
    """Build a row-stochastic transition-probability matrix from embeddings.

    Args:
        z: ``(N, d)`` embeddings.
        kernel_method: ``"gaussian"`` or ``"tstudent"``.
        alpha: Exponent for the Gaussian kernel (controls tail weight).
        bandwidth: Length-scale for the Gaussian kernel.

    Returns:
        ``(N, N)`` row-stochastic matrix (rows sum to 1).
    """
    if kernel_method == "gaussian":
        dist = torch.cdist(z, z, p=2)
        k = torch.exp(-((dist / bandwidth) ** alpha))
        row_sum = torch.sum(k, dim=1, keepdim=True)
        return k / row_sum
    elif kernel_method == "tstudent":
        dist_sq = torch.cdist(z, z, p=2) ** 2
        numerator = (1.0 + dist_sq) ** (-1.0)
        row_sum = torch.sum(numerator, dim=1, keepdim=True)
        return numerator / row_sum
    else:
        raise ValueError(
            f"Unknown kernel_method: {kernel_method!r}. Use 'gaussian' or 'tstudent'."
        )
