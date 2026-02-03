"""Activation extraction via PyTorch forward hooks."""
from dataclasses import dataclass
from typing import Literal

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
