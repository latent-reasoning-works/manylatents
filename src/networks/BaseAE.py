import torch
from src.networks.MLP import MLP

class BaseAE(torch.nn.Module):
    def __init__(self, dim, emb_dim, width=64, depth = 4, activation_fn=torch.nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim

        self.encoder = MLP(dim, emb_dim, width=width, depth = depth, activation_fn=activation_fn)
        self.decoder = MLP(emb_dim, dim, width=width, depth = depth, activation_fn=activation_fn)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))