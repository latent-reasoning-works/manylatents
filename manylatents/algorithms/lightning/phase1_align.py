"""Phase1 alignment pre-training helper.

Imperative loop that runs a student against a frozen
:class:`~manylatents.lightning.activation_snapshot.ActivationSnapshot` for a
fixed number of optimizer steps, minimizing weighted MSE between the
student's aligned-layer activations and the snapshot's stored targets.

Deliberately NOT a ``trainer.fit`` call and NOT a LightningModule method.
Phase1 is small, closed-form against a fixed buffer, and doesn't need
Lightning's dataloader-per-step machinery. Exposing it as a free function
taking any ``nn.Module`` keeps it reusable beyond
:class:`~manylatents.algorithms.lightning.distillation.Distillation` - probing
warmups, CKA warmups, and other non-Lightning scenarios can call this
directly.

See the plan at ``/network/scratch/c/cesar.valdez/distillation-algo-module-plan.md``
for the architectural commitments driving this split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from manylatents.lightning.activation_snapshot import ActivationSnapshot
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec, resolve_layer

__all__ = ["align_on_snapshot"]


def _device_matches(tensor_device: torch.device, expected: torch.device) -> bool:
    """Compare devices respecting torch's "unspecified index" semantic.

    ``torch.device("cuda")`` has ``index=None`` and should match *any* cuda
    index. ``torch.device("cuda:0")`` only matches cuda device 0. Same for
    cpu/meta/mps. This matches how PyTorch itself handles missing indices in
    ``Tensor.to(device)``.
    """
    if tensor_device.type != expected.type:
        return False
    if expected.index is None:
        return True
    return tensor_device.index == expected.index


def _assert_snapshot_on_device(snap: ActivationSnapshot, device: torch.device) -> None:
    """Raise ValueError if any snapshot tensor is on a different device.

    Frozen snapshots cannot be mutated in place (``@dataclass(frozen=True)``),
    so a silent ``.to(device)`` would force us to return a new snapshot - which
    breaks caller expectations that the object they passed in is the object
    being used. Instead we refuse and make device management the caller's
    responsibility. This matches the plan's "no silent copies" rule.
    """
    tensors: List[tuple] = [
        ("input_ids", snap.input_ids),
        ("attention_mask", snap.attention_mask),
    ]
    for path, acts in snap.activations.items():
        tensors.append((f"activations[{path!r}]", acts))

    mismatches = [
        (name, t.device) for name, t in tensors
        if not _device_matches(t.device, device)
    ]
    if mismatches:
        raise ValueError(
            f"align_on_snapshot received device={device} but the following "
            f"snapshot tensors are on other devices: {mismatches}. Load the "
            f"snapshot onto the target device before calling; we refuse to "
            f"mutate a frozen dataclass silently."
        )


def align_on_snapshot(
    student: nn.Module,
    snapshot: ActivationSnapshot,
    layer_pairs: List[Mapping[str, Any]],
    n_steps: int,
    optimizer_cfg: Mapping[str, Any],
    *,
    batch_size: int = 16,
    seed: int = 42,
    device: str = "cuda",
) -> List[float]:
    """Run ``n_steps`` optimizer steps of alignment-only SGD on the student.

    Fresh AdamW built from ``optimizer_cfg`` (``learning_rate``/``lr``,
    ``weight_decay``, ``betas``, ``eps``). Does not share optimizer state with
    any phase2/3 optimizer the caller may build later. Returns the per-step
    loss trajectory as a list of floats.

    Args:
        student: model being aligned. Moved to ``device`` by this function.
        snapshot: frozen reference of pre-pooled activations. Its tensors
            must already be on ``device``; this function will not move them.
        layer_pairs: list of
            ``{"student": <student_layer_path>, "teacher": <snapshot_key>, "weight": <float>}``.
        n_steps: number of optimizer steps to run. Returns a list of length
            ``n_steps`` (or empty if ``n_steps == 0``).
        optimizer_cfg: AdamW hyperparameters. Keys: ``learning_rate``/``lr``,
            ``weight_decay``, ``betas``, ``eps``.
        batch_size: probe minibatch size. Capped at ``len(snapshot)``.
        seed: used to seed ``torch.Generator`` for per-step probe sampling.
            For per-trial variation in sweeps, pass the same ``seed`` that
            drives the consumer's datamodule (``TextDataModule.seed`` or
            equivalent).
        device: where the student and snapshot tensors must live.

    Returns:
        List of per-step weighted-MSE losses (detached floats).

    Raises:
        ValueError: if any snapshot tensor is on a device other than
            ``device``, or if a student layer path does not resolve.
    """
    if n_steps == 0:
        return []

    torch_device = torch.device(device)
    _assert_snapshot_on_device(snapshot, torch_device)
    student = student.to(torch_device)

    # Validate student paths before starting the loop so misconfig fails fast.
    for pair in layer_pairs:
        try:
            resolve_layer(student, pair["student"])
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"layer_pairs student={pair['student']!r} does not resolve "
                f"on the provided student module: {exc}"
            ) from exc

    # AdamW (no split into decay/no-decay groups; phase1 is short enough that
    # the distinction rarely affects the finding, and avoiding the split keeps
    # this helper simple and independent of Distillation's internals).
    lr = float(optimizer_cfg.get("learning_rate", optimizer_cfg.get("lr", 1e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    eps = float(optimizer_cfg.get("eps", 1e-8))
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    specs = [
        LayerSpec(path=pair["student"], reduce=snapshot.reduction)
        for pair in layer_pairs
    ]
    extractor = ActivationExtractor(specs, detach=False)

    n_probe = len(snapshot)
    k = min(batch_size, n_probe)

    prior_training = student.training
    student.train()
    losses: List[float] = []
    try:
        # One capture context wraps the whole loop: hooks installed once,
        # removed on exit. Activations cleared per step via get_activations().
        with extractor.capture(student):
            for step in range(n_steps):
                gen = torch.Generator(device=torch_device).manual_seed(seed + step)
                idx = torch.randperm(n_probe, device=torch_device, generator=gen)[:k]

                probe_ids = snapshot.input_ids[idx]
                probe_mask = snapshot.attention_mask[idx]

                optimizer.zero_grad()
                student(input_ids=probe_ids, attention_mask=probe_mask)
                student_acts = extractor.get_activations(clear=True)

                total = torch.zeros((), device=torch_device, dtype=torch.float32)
                for pair in layer_pairs:
                    w = float(pair.get("weight", 1.0))
                    target = snapshot.activations[pair["teacher"]][idx]
                    live = student_acts[pair["student"]]
                    total = total + w * ((live - target.to(live.dtype)) ** 2).mean()

                total.backward()
                optimizer.step()
                losses.append(float(total.detach().cpu()))
    finally:
        if not prior_training:
            student.eval()

    return losses
