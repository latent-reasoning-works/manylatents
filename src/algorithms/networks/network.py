from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class HasForward(Protocol):
    def forward(self, x: torch.Tensor, **kwargs) -> Any: ...