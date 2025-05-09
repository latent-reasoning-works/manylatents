import torch.nn as nn


class MSELoss(nn.Module):
    def forward(self, outputs, targets, **kwargs):
        return nn.functional.mse_loss(outputs, targets)
