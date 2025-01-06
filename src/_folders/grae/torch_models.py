import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class LinearActivation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

class LinearBlock(nn.Sequential):
    def __init__(self, dim_list):
        modules = [LinearActivation(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))
        super().__init__(*modules)

class AETorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()

        full_list = [input_dim] + list(hidden_dims) + [z_dim]
        self.encoder = LinearBlock(dim_list=full_list)

        full_list.reverse()
        full_list[0] = z_dim
        self.decoder = LinearBlock(dim_list=full_list)

    def forward(self, x):
        z = self.encoder(x)
        z_decoder = z
        recon = self.decoder(z_decoder)
        return recon, z

class ProxAETorchModule(AETorchModule):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__(input_dim, hidden_dims, z_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Log Softmax Output Layer for numerical stability with nn.KLDivLoss

    def forward(self, x):
        z = self.encoder(x)
        z_decoder = z
        recon = self.decoder(z_decoder)
        recon = self.log_softmax(recon)  # Apply log softmax activation to the output
        return recon, z


class MLPReg(nn.Module):
    def __init__(self, encoder, input_dim, output_dim):
        super().__init__()

        self.encoder = encoder
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.linear(z)
        return pred
