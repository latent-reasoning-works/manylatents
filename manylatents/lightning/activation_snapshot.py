"""Frozen per-layer activation snapshot.

An ActivationSnapshot pairs a fixed set of tokenized inputs with pre-computed,
already-pooled activations at named layers. Used as a reference artifact for
alignment-style training losses (see
``manylatents.algorithms.lightning.distillation``), probing, and analysis that
requires stable target activations against a known input set.

The snapshot declares, via the ``reduction`` field, how its per-layer tensors
were pooled (``mean``, ``cls``, ``last_token``, ``first_token``, ``none``).
Consumers that perform a fresh student forward for a regularizer use this field
to drive matching pooling of the live activations; they do not re-state
producer-side recipes.

Contract for ``sample_ids``: when a snapshot is built against a probe split
from a consumer's ``LightningDataModule`` (e.g. ``TextDataModule``), its
``sample_ids`` must be the same identifiers the datamodule emits at training
time (e.g. ``batch["probe_ids"]``). Mismatched ID spaces cause silent
target-lookup errors. This is a consumer-side convention; we enforce uniqueness
on the snapshot side and document the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from manylatents.lightning.hooks import (
    VALID_REDUCE,
    ActivationExtractor,
    LayerSpec,
)

__all__ = ["ActivationSnapshot", "SNAPSHOT_SCHEMA_VERSION"]

# Bump when the on-disk format changes incompatibly. `load` rejects unknown
# versions rather than silently mis-deserializing. Migration paths (if any) go
# inside `load` keyed on the version field.
SNAPSHOT_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class ActivationSnapshot:
    """Fixed batch of tokenized inputs with pre-computed, already-pooled
    activations at named layers.

    Attributes:
        input_ids: (N, L) int64 token IDs.
        attention_mask: (N, L) int64 attention mask.
        sample_ids: length-N list of unique identifiers (see contract above).
        activations: dict mapping layer path (str) to a (N, hidden) tensor of
            activations already pooled with ``reduction``.
        reduction: the single pooling strategy applied to every tensor in
            ``activations``. One of the entries in
            ``manylatents.lightning.hooks.VALID_REDUCE``.
    """

    input_ids: Tensor
    attention_mask: Tensor
    sample_ids: List[int]
    activations: Dict[str, Tensor]
    reduction: str

    def __post_init__(self) -> None:
        n = self.input_ids.shape[0]

        if self.attention_mask.shape[0] != n:
            raise ValueError(
                f"attention_mask.shape[0] ({self.attention_mask.shape[0]}) "
                f"must equal input_ids.shape[0] ({n})"
            )
        if len(self.sample_ids) != n:
            raise ValueError(
                f"len(sample_ids) ({len(self.sample_ids)}) must equal "
                f"input_ids.shape[0] ({n})"
            )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")

        if self.reduction not in VALID_REDUCE:
            raise ValueError(
                f"reduction must be one of {VALID_REDUCE}, got {self.reduction!r}"
            )

        input_device = self.input_ids.device
        if self.attention_mask.device != input_device:
            raise ValueError(
                f"attention_mask device ({self.attention_mask.device}) must "
                f"equal input_ids device ({input_device})"
            )

        for layer, acts in self.activations.items():
            if acts.shape[0] != n:
                raise ValueError(
                    f"activations[{layer!r}].shape[0] ({acts.shape[0]}) must "
                    f"equal input_ids.shape[0] ({n})"
                )
            if acts.device != input_device:
                raise ValueError(
                    f"activations[{layer!r}] device ({acts.device}) must "
                    f"equal input_ids device ({input_device})"
                )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def save(self, path: Path | str) -> None:
        """Serialize to disk as a single ``torch.save`` blob.

        The on-disk format is a versioned dict. ``load`` validates the version
        and reconstructs the dataclass (re-triggering ``__post_init__``).
        """
        torch.save(
            {
                "_version": SNAPSHOT_SCHEMA_VERSION,
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "sample_ids": list(self.sample_ids),
                "activations": dict(self.activations),
                "reduction": self.reduction,
            },
            str(path),
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
        sample_ids: Sequence[int],
        layer_paths: Sequence[str],
        *,
        reduction: str = "mean",
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> "ActivationSnapshot":
        """Run ``model`` over ``(input_ids, attention_mask)`` and capture
        activations at every path in ``layer_paths`` using ``reduction``.

        This is the bridge between :class:`ActivationExtractor` (per-step hook
        machinery) and a frozen snapshot. Model is run in eval + no_grad; hooks
        are installed for exactly the duration of the batch loop and removed
        on exit (via the extractor's context manager).

        Args:
            model: the model to forward. Will be put in eval mode for the call
                and restored to its prior training mode on return.
            input_ids: (N, L) long tensor of token ids.
            attention_mask: (N, L) long tensor of attention mask values.
            sample_ids: length-N sequence of unique integer ids. See the
                class docstring for the consumer-side contract.
            layer_paths: dotted paths to capture activations at. Resolution
                uses :func:`manylatents.lightning.hooks.resolve_layer`, which
                handles HF-family aliases (transformer.h <-> gpt_neox.layers
                <-> model.layers). Every path must resolve to a module on
                ``model``.
            reduction: single pooling strategy applied to every captured
                tensor. Must be one of ``VALID_REDUCE``. Per-layer reductions
                are deliberately not supported — build multiple snapshots.
            batch_size: forward-pass batch size. Does not change the result.
            device: if given, move ``model``, ``input_ids``, and
                ``attention_mask`` to this device before running. Outputs end
                up on ``device``. If ``None``, runs wherever the inputs are.

        Returns:
            A frozen :class:`ActivationSnapshot` carrying the pooled
            activations keyed by layer path.
        """
        if reduction not in VALID_REDUCE:
            raise ValueError(
                f"reduction must be one of {VALID_REDUCE}, got {reduction!r}"
            )

        n = input_ids.shape[0]
        if attention_mask.shape[0] != n:
            raise ValueError(
                f"attention_mask.shape[0] ({attention_mask.shape[0]}) must "
                f"equal input_ids.shape[0] ({n})"
            )
        if len(sample_ids) != n:
            raise ValueError(
                f"len(sample_ids) ({len(sample_ids)}) must equal "
                f"input_ids.shape[0] ({n})"
            )

        prior_training = model.training
        model.eval()
        try:
            if device is not None:
                model = model.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            specs = [LayerSpec(path=p, reduce=reduction) for p in layer_paths]
            extractor = ActivationExtractor(specs, detach=True)

            with torch.no_grad():
                with extractor.capture(model):
                    for start in range(0, n, batch_size):
                        end = min(start + batch_size, n)
                        model(
                            input_ids=input_ids[start:end],
                            attention_mask=attention_mask[start:end],
                        )

            activations = extractor.get_activations()
        finally:
            if prior_training:
                model.train()

        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_ids=list(sample_ids),
            activations=activations,
            reduction=reduction,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ActivationSnapshot":
        """Load a snapshot previously written by ``save``.

        Raises:
            ValueError: if the file's ``_version`` does not match
                ``SNAPSHOT_SCHEMA_VERSION`` or the dict is missing required keys.
        """
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            raise ValueError(
                f"expected a dict at {path!s}, got {type(blob).__name__}"
            )
        version = blob.get("_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unknown ActivationSnapshot schema _version={version!r} at "
                f"{path!s}; this build expects {SNAPSHOT_SCHEMA_VERSION}"
            )
        required = {"input_ids", "attention_mask", "sample_ids", "activations", "reduction"}
        missing = required - blob.keys()
        if missing:
            raise ValueError(
                f"malformed snapshot at {path!s}: missing keys {sorted(missing)}"
            )
        return cls(
            input_ids=blob["input_ids"],
            attention_mask=blob["attention_mask"],
            sample_ids=list(blob["sample_ids"]),
            activations=dict(blob["activations"]),
            reduction=blob["reduction"],
        )
