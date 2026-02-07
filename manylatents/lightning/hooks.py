"""Activation extraction via PyTorch forward hooks."""
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

VALID_REDUCE = ("mean", "last_token", "cls", "first_token", "all", "none")
VALID_EXTRACTION_POINTS = ("output", "input", "hidden_states")


@dataclass
class LayerSpec:
    """Specification for which layer to extract activations from.

    Attributes:
        path: Dot-separated path to layer, e.g., "model.layers[-1]"
        extraction_point: What to extract - "output", "input", or "hidden_states"
        reduce: How to reduce sequence dimension:
            - "mean": Average over sequence
            - "last_token": Take last token
            - "cls" / "first_token": Take first token (CLS)
            - "all": Keep full sequence (N, seq_len, dim)
            - "none": No reduction, return raw
    """
    path: str
    extraction_point: Literal["output", "input", "hidden_states"] = "output"
    reduce: Literal["mean", "last_token", "cls", "first_token", "all", "none"] = "mean"

    def __post_init__(self):
        if self.reduce not in VALID_REDUCE:
            raise ValueError(
                f"reduce must be one of {VALID_REDUCE}, got '{self.reduce}'"
            )
        if self.extraction_point not in VALID_EXTRACTION_POINTS:
            raise ValueError(
                f"extraction_point must be one of {VALID_EXTRACTION_POINTS}, "
                f"got '{self.extraction_point}'"
            )


def resolve_layer(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated path to a layer in a model.

    Supports:
    - Dot notation: "layers.0.self_attn"
    - Index notation: "layers[0].self_attn"
    - Negative indices: "layers[-1]"

    Args:
        model: The model to traverse
        path: Dot-separated path with optional bracket indices

    Returns:
        The resolved layer

    Raises:
        AttributeError: If path component doesn't exist
        IndexError: If index is out of bounds
    """
    current: Any = model

    # Split on dots, but preserve bracket notation
    # "layers[-1].self_attn" -> ["layers[-1]", "self_attn"]
    parts = re.split(r'\.(?![^\[]*\])', path)

    for part in parts:
        # Check for index notation: "layers[0]" or "layers[-1]"
        match = re.match(r'(\w+)\[(-?\d+)\]', part)
        if match:
            attr_name, idx = match.groups()
            current = getattr(current, attr_name)
            current = current[int(idx)]
        else:
            if not hasattr(current, part):
                raise AttributeError(
                    f"'{type(current).__name__}' has no attribute '{part}'"
                )
            current = getattr(current, part)

    return current


class ActivationExtractor:
    """Extract activations from specified layers using forward hooks.

    Usage:
        extractor = ActivationExtractor([LayerSpec("model.layers[-1]")])
        with extractor.capture(model):
            model(inputs)
        activations = extractor.get_activations()
    """

    def __init__(self, layer_specs: List[LayerSpec]):
        self.layer_specs = layer_specs
        self._activations: Dict[str, List[Tensor]] = {}
        self._handles: List = []

    def _make_hook(self, spec: LayerSpec):
        """Create a forward hook for the given spec."""
        def hook(module, input, output):
            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Reduce sequence dimension if needed
            tensor = self._reduce(tensor, spec.reduce)

            if spec.path not in self._activations:
                self._activations[spec.path] = []
            self._activations[spec.path].append(tensor.detach())

        return hook

    def _reduce(self, tensor: Tensor, method: str) -> Tensor:
        """Reduce sequence dimension."""
        if method == "none" or tensor.dim() < 3:
            return tensor

        # Assume shape is (batch, seq_len, dim)
        if method == "mean":
            return tensor.mean(dim=1)
        elif method == "last_token":
            return tensor[:, -1, :]
        elif method in ("cls", "first_token"):
            return tensor[:, 0, :]
        elif method == "all":
            return tensor
        else:
            return tensor

    @contextmanager
    def capture(self, model: nn.Module):
        """Context manager to capture activations during forward pass."""
        # Register hooks
        for spec in self.layer_specs:
            layer = resolve_layer(model, spec.path)
            handle = layer.register_forward_hook(self._make_hook(spec))
            self._handles.append(handle)

        try:
            yield self
        finally:
            # Remove hooks
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    def get_activations(self, clear: bool = True) -> Dict[str, Tensor]:
        """Get captured activations, concatenated over batches.

        Args:
            clear: Whether to clear activations after getting them

        Returns:
            Dict mapping layer path to activation tensor
        """
        result = {}
        for path, tensors in self._activations.items():
            if tensors:
                result[path] = torch.cat(tensors, dim=0)

        if clear:
            self._activations.clear()

        return result

    def clear(self):
        """Clear stored activations."""
        self._activations.clear()
