# src/networks/aanet/base.py

from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Base(nn.Module):
    """
    Base class for AAnet variants. Implements functions to calculate barycentric coordinates,
    and provides utility methods to translate between the archetypal and feature spaces.
    
    # built to mimic https://github.com/KrishnaswamyLab/AAnet/blob/master/AAnet_torch/models/AAnet_base.py
    """
    def __init__(self) -> None:
        super(Base, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def get_archetypes_latent(self) -> Tensor:
        return torch.tensor(
            np.vstack([torch.eye(self.n_archetypes - 1), np.zeros(self.n_archetypes - 1)])
        )

    def get_n_simplex(self, n: int = 2, scale: int = 1) -> Tensor:
        """
        Returns an n-simplex centered at the origin in the feature space.
        """
        nth = 1 / (n - 1) * (1 - np.sqrt(n)) * np.ones(n - 1)
        D = np.vstack([np.eye(n - 1), nth]) * scale
        return torch.tensor(D - np.mean(D, axis=0), dtype=torch.float, device=self.device)

    def get_archetypes_data(self) -> Any:
        """Returns archetypes in the feature domain."""
        return self.decode(self.get_n_simplex(self.n_archetypes, self.simplex_scale))

    def euclidean_to_barycentric(self, X: Tensor) -> Tensor:
        """
        Converts Euclidean coordinates to barycentric coordinates with respect to the archetypal simplex.
        Requires `self.archetypal_simplex` to be set.
        """
        simplex = self.archetypal_simplex
        T = torch.zeros((X.shape[1], X.shape[1]), device=self.device)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                T[i, j] = simplex[i, j] - simplex[-1, j]
        T_inv = torch.inverse(T.float()).to(self.device)
        X_bary = torch.einsum('ij,bj->bi', T_inv, X - simplex[-1])
        X_bary = torch.cat([X_bary, (1 - torch.sum(X_bary, dim=1, keepdim=True))], dim=1)
        return X_bary

    def is_in_simplex(self, X_bary: Tensor) -> Tensor:
        """Returns True for points inside the simplex, False otherwise."""
        all_non_negative = torch.sum(X_bary >= 0, dim=1) == X_bary.shape[1]
        all_convex = torch.sum(X_bary <= 1, dim=1) == X_bary.shape[1]
        return all_non_negative & all_convex

    def dist_to_simplex(self, X_bary: Tensor) -> Tensor:
        """
        Sums all negative values outside the simplex.
        (Note: may yield lower loss values on the boundaries of the Voronoi regions.)
        """
        return torch.sum(
            torch.where(X_bary < 0, torch.abs(X_bary), torch.zeros_like(X_bary)),
            dim=1
        ).to(self.device)

    def calc_archetypal_loss(self, archetypal_embedding: Tensor) -> Tensor:
        """
        Returns MSE archetypal loss (sum of negative values outside the simplex).
        """
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean(self.dist_to_simplex(X_bary) ** 2)
    
    def calc_diffusion_extrema_loss(self, archetypal_embedding: Tensor) -> Tensor:
        """
        Returns MSE diffusion extrema loss: minimize MSE between diffusion extrema and archetypes.
        Diffusion extrema are assumed to be the first n_archetypes samples in the batch.
        """
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean((X_bary[:self.n_archetypes, :] - torch.eye(self.n_archetypes).to(self.device)) ** 2)

class Vanilla(Base):
    """
    AAnet vanilla variant that implements a simple autoencoder-style network
    without the variational (KL divergence) components.
    
    built to mimic https://github.com/KrishnaswamyLab/AAnet/blob/master/AAnet_torch/models/AAnet_vanilla.py
    """
    def __init__(
        self,
        input_shape: int,
        n_archetypes: int = 4,
        noise: float = 0,
        layer_widths: list = [128, 128],
        activation_out: str = "tanh",
        simplex_scale: int = 1,
        device: torch.device = None,
        diffusion_extrema=None,
        **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.n_archetypes = n_archetypes
        self.noise = noise
        self.layer_widths = layer_widths
        self.activation_out = activation_out
        self.simplex_scale = simplex_scale
        self.diffusion_extrema = diffusion_extrema
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build encoder layers.
        encoder_layers = []
        in_dim = input_shape
        for i, width in enumerate(layer_widths):
            if i == 0:
                encoder_layers.append(nn.Linear(in_features=in_dim, out_features=width))
            else:
                encoder_layers.append(nn.Linear(in_features=layer_widths[i - 1], out_features=width))
        # Final encoder layer (produces the latent embedding).
        encoder_layers.append(nn.Linear(in_features=layer_widths[-1], out_features=n_archetypes - 1))
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Build decoder layers.
        decoder_layers = []
        dec_widths = layer_widths[::-1]
        in_dim = n_archetypes - 1
        for i, width in enumerate(dec_widths):
            if i == 0:
                decoder_layers.append(nn.Linear(in_features=in_dim, out_features=width))
            else:
                decoder_layers.append(nn.Linear(in_features=dec_widths[i - 1], out_features=width))
        # Final decoder layer.
        decoder_layers.append(nn.Linear(in_features=dec_widths[-1], out_features=input_shape))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Precompute the archetypal simplex.
        self.archetypal_simplex = self.get_n_simplex(self.n_archetypes, scale=self.simplex_scale)
        self.to(self.device)

    def encode(self, features: Tensor) -> Tensor:
        activation = features
        for i, layer in enumerate(self.encoder_layers[:-1]):
            activation = torch.relu(layer(activation))
        # No activation for final encoder layer.
        activation = self.encoder_layers[-1](activation)
        return activation

    def decode(self, activation: Tensor) -> Tensor:
        for layer in self.decoder_layers[:-1]:
            activation = torch.relu(layer(activation))
        # Final decoder layer (no activation).
        activation = self.decoder_layers[-1](activation)
        return activation

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            recons: reconstructed input,
            input: original input,
            archetypal_embedding: latent embedding used for computing archetypal loss.
        """
        # Encode the input.
        activation = self.encode(input)
        archetypal_embedding = activation.clone()
        # Optionally add noise.
        if self.noise > 0:
            activation += torch.normal(mean=0., std=self.noise, size=activation.shape)
        # Decode the latent representation.
        recons = self.decode(activation)
        return recons, input, archetypal_embedding

    def loss_function(self, outputs: Tuple[Tensor, Tensor, Tensor], **kwargs) -> Tensor:
        recons, input, embedding = outputs
        mse_loss = F.mse_loss(recons, input)
        archetypal_loss = self.calc_archetypal_loss(embedding)
        return mse_loss + self.archetypal_weight * archetypal_loss

class VAE(Base):
    """
    Implements AAnet as a Variational Autoencoder (VAE) for reconstruction,
    adding noise in the latent space via reparameterization.
    
    https://github.com/KrishnaswamyLab/AAnet/blob/master/AAnet_torch/models/AAnet_VAE.py
    """
    def __init__(
        self,
        input_shape: int,
        n_archetypes: int = 4,
        layer_widths: List[int] = [128, 128],
        activation_out: str = "tanh",
        simplex_scale: int = 1,
        archetypal_weight: float = 1,
        kl_loss: str = "partial",
        device: torch.device = None,
        diffusion_extrema=None,
        **kwargs
    ) -> None:
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.n_archetypes = n_archetypes
        self.layer_widths = layer_widths
        self.activation_out = activation_out.lower()
        self.simplex_scale = simplex_scale
        self.archetypal_weight = archetypal_weight
        self.kl_loss = kl_loss
        self.diffusion_extrema = diffusion_extrema
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build encoder.
        layers = []
        in_dim = input_shape
        for width in layer_widths:
            layers.append(nn.Sequential(nn.Linear(in_dim, width), nn.ReLU()))
            in_dim = width
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(layer_widths[-1], n_archetypes - 1)
        self.fc_var = nn.Linear(layer_widths[-1], n_archetypes - 1)
        
        # Build decoder.
        self.decoder_input = nn.Linear(n_archetypes - 1, layer_widths[-1])
        decoder_layers = []
        dec_widths = layer_widths[::-1]
        for i in range(len(dec_widths) - 1):
            decoder_layers.append(nn.Sequential(
                nn.Linear(dec_widths[i], dec_widths[i + 1]),
                nn.ReLU(),
            ))
        self.decoder = nn.Sequential(*decoder_layers)
        if self.activation_out == 'tanh':
            act_out = nn.Tanh()
        elif self.activation_out in ["linear", None]:
            act_out = None
        else:
            raise ValueError('activation_out not recognized')
        self.final_layer = nn.Sequential(nn.Linear(dec_widths[-1], input_shape), act_out)
        
        # Precompute the archetypal simplex.
        self.archetypal_simplex = self.get_n_simplex(self.n_archetypes, scale=self.simplex_scale)
        self.to(self.device)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing it through the encoder network,
        then splits the result into mean and log-variance components.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the latent code z to the reconstructed input.
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        archetypal_embedding = mu.clone()  # Use mu for archetypal loss
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        # Return outputs needed for loss calculation.
        return [recons, input, archetypal_embedding, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss including reconstruction, KL divergence, and archetypal losses.
        Expects kwargs to contain 'M_N' for minibatch weighting.
        """
        outputs = args
        recons = outputs[0]
        input = outputs[1]
        mu = outputs[2]
        log_var = outputs[3]
        kld_weight = kwargs.get('M_N', 1.0)
        
        recons_loss = F.mse_loss(recons, input)
        if self.kl_loss is False or self.kl_loss is None:
            kld_loss = 0
        elif self.kl_loss.lower() == "partial":
            kld_loss = torch.mean(torch.sum((1 - log_var.exp()) ** 2, dim=1))
        elif self.kl_loss.lower() == "full":
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        else:
            raise ValueError("`kl_loss` must be either 'partial' or 'full'")
        
        archetypal_loss = self.calc_archetypal_loss(mu)
        loss = recons_loss + kld_weight * kld_loss + self.archetypal_weight * archetypal_loss
        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss, 'Archetypal_Loss': archetypal_loss}
