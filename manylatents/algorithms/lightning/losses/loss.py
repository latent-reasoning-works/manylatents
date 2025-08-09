from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class FunctionLoss(Protocol):
    """
    Protocol for loss functions in our framework.
    Named as such [YourFunction]Loss to avoid confusion 
    with PyTorch's built-in loss functions.

    A Loss function should be callable or nn.Module-like, accepting:
      - outputs: model predictions or reconstructions
      - targets: ground truth tensor
      - additional keyword-only extras
    and returning a scalar torch.Tensor.
    """
    def __call__(
        self,
        outputs: Tensor,
        targets: Tensor,
        **extras: object,
    ) -> Tensor:
        ...  # Implement loss computation
