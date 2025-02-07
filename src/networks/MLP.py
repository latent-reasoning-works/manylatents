import torch

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, width=64, depth=4, activation_fn=torch.nn.ReLU()):
        super().__init__()
        if out_dim is None:
            out_dim = dim // 2

        layers = [torch.nn.Linear(dim, width), activation_fn]  # Input layer

        for _ in range(depth - 2):  # Hidden layers
            layers.append(torch.nn.Linear(width, width))
            layers.append(activation_fn)

        layers.append(torch.nn.Linear(width, out_dim))  # Output layer

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)