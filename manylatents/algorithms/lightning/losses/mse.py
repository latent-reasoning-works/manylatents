import torch.nn as nn
from torch import Tensor


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs: Tensor, targets: Tensor, **kwargs) -> Tensor:
         return self.mse(outputs, targets)
