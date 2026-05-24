"""Reusable contracts for model representations.

Plain functions — no pytest, no class hierarchy — so they compose anywhere: in
unit tests, as runtime assertions on real data, or in a notebook, and across
manylatents and its extensions (e.g. manylatents.omics encoders).

The headline guard is the "overfit one batch" sanity check for per-layer
extraction: requesting N layers must yield N *distinct* embeddings. The bug
this catches produced N identical layers (every layer a copy of one tensor),
which is invisible unless something asserts the layers actually differ.
"""
from __future__ import annotations

from typing import Callable, Mapping

import torch
from torch import Tensor

__all__ = ["assert_layers_distinct", "assert_per_layer_contract"]


def _stack(per_layer: Mapping[object, Tensor]) -> Tensor:
    """{key: (...,)} → (n_layers, ...) stacked, detached, float, on CPU."""
    return torch.stack([per_layer[k].detach().float().cpu() for k in per_layer])


def assert_layers_distinct(
    per_layer: Mapping[object, Tensor], *, atol: float = 1e-6,
) -> None:
    """Assert a per-layer embedding mapping has not collapsed — i.e. the layers
    are not identical copies of a single tensor.

    Args:
        per_layer: {layer_key: tensor} as returned by a multi-layer encoder.
        atol: max cross-layer std below which layers are treated as identical.

    Raises:
        AssertionError: if every requested layer is the same within `atol`.
    """
    keys = list(per_layer)
    if len(keys) < 2:
        return  # nothing to compare
    spread = _stack(per_layer).std(dim=0)  # per-element std across layers
    if float(spread.max()) <= atol:
        raise AssertionError(
            f"layer collapse: {len(keys)} layers are identical within "
            f"atol={atol} (max cross-layer std {float(spread.max()):.2e}). "
            f"Layer keys: {keys}"
        )


def assert_per_layer_contract(
    encode_fn: Callable[[object], Mapping[object, Tensor]],
    input_a: object,
    input_b: object,
    *,
    atol: float = 1e-6,
) -> None:
    """Full per-layer encoder sanity check on a tiny batch:

      1. encode_fn(input_a) returns a non-empty per-layer mapping;
      2. its layers are distinct (assert_layers_distinct);
      3. a different input (input_b) yields different embeddings.

    `encode_fn` must return a Mapping[layer_key -> Tensor].

    Raises:
        AssertionError: on an empty/non-mapping return, layer collapse, or an
            encoder that produces identical embeddings for the two inputs.
    """
    out_a = encode_fn(input_a)
    if not isinstance(out_a, Mapping) or not out_a:
        raise AssertionError(
            "encode_fn must return a non-empty per-layer mapping; "
            f"got {type(out_a).__name__}"
        )
    assert_layers_distinct(out_a, atol=atol)

    out_b = encode_fn(input_b)
    shared = [k for k in out_a if k in out_b]
    if shared and all(
        torch.allclose(
            out_a[k].detach().float(), out_b[k].detach().float(), atol=atol
        )
        for k in shared
    ):
        raise AssertionError(
            "encoder produced identical embeddings for two different inputs — "
            "it is not distinguishing inputs"
        )
