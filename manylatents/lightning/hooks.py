"""Activation extraction via PyTorch forward hooks."""
import re
from dataclasses import dataclass
from typing import Any, Literal

import torch.nn as nn

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
