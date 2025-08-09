import torch

from src.algorithms.losses.mse import MSELoss


def test_loss_runs_and_returns_scalar():
    loss = MSELoss()
    outputs = torch.randn(2, 3)
    targets = torch.randn(2, 3)
    # pass an extra kwarg to prove **extras is consumed
    val = loss(outputs, targets, unused_kwarg=42)
    assert isinstance(val, torch.Tensor)
    assert val.ndim == 0
