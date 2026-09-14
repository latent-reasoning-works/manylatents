"""Opt-in debug callbacks: NaN/Inf detector and first-batch logger.

Both surfaces are imported by callers that gate them on env vars (e.g.
``DEBUG_NAN_HOOKS=1``). Nothing here is wired into manylatents core flows.
"""
from __future__ import annotations

import math
import os
from numbers import Integral
from typing import Any, List

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.utils.hooks import RemovableHandle

from manylatents.utils.exceptions import MeasurementUnavailable


def _validate_count(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _loss_summary(loss: Any) -> str:
    if isinstance(loss, torch.Tensor) and loss.numel() != 1:
        return str(MeasurementUnavailable("loss must contain one scalar").to_dict())
    try:
        value = float(loss)
    except (TypeError, ValueError, OverflowError) as exc:
        return str(MeasurementUnavailable(f"loss cannot be read as a scalar: {exc}").to_dict())
    if not math.isfinite(value):
        return str(MeasurementUnavailable("loss is non-finite").to_dict())
    return f"{value:.6e}"


def attach_nan_detector(model: nn.Module, max_reports: int = 10) -> List[RemovableHandle]:
    """Register forward hooks on every leaf module that print the first
    ``max_reports`` NaN/Inf occurrences in outputs and then go silent.

    Returns the list of handles so callers can remove them later.
    """
    _validate_count("max_reports", max_reports)
    rank = os.environ.get("LOCAL_RANK", "0")
    state = {"reports": 0, "max": max_reports}
    handles: List[RemovableHandle] = []

    def make_hook(name: str):
        def hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
            if state["reports"] >= state["max"]:
                return
            if isinstance(out, torch.Tensor):
                tensors = [(name, out)]
            elif isinstance(out, (tuple, list)):
                tensors = [(f"{name}[{i}]", t) for i, t in enumerate(out)
                           if isinstance(t, torch.Tensor)]
            else:
                return
            for tname, t in tensors:
                if not torch.is_floating_point(t):
                    continue
                nan_mask = torch.isnan(t)
                inf_mask = torch.isinf(t)
                has_nan = bool(nan_mask.any().item())
                has_inf = bool(inf_mask.any().item())
                if not (has_nan or has_inf):
                    continue
                finite_mask = ~nan_mask & ~inf_mask
                n_finite = int(finite_mask.sum().item())
                if n_finite:
                    amax = f"{t[finite_mask].abs().max().item():.4e}"
                else:
                    amax = str(MeasurementUnavailable("no finite activations").to_dict())
                n_nan = int(nan_mask.sum().item())
                n_inf = int(inf_mask.sum().item())
                print(
                    f"[NaN-trap][rank{rank}] FIRST_BAD module={tname!r} "
                    f"shape={tuple(t.shape)} dtype={t.dtype} "
                    f"NaN={n_nan} Inf={n_inf} "
                    f"finite_count={n_finite} total_count={t.numel()} "
                    f"finite_abs_max={amax}",
                    flush=True,
                )
                state["reports"] += 1
                if state["reports"] >= state["max"]:
                    print(
                        f"[NaN-trap][rank{rank}] reached max_reports={state['max']}, "
                        f"further bad outputs will be ignored",
                        flush=True,
                    )
                return

        return hook

    for name, m in model.named_modules():
        if any(True for _ in m.children()):
            continue
        h = m.register_forward_hook(make_hook(name or "<root-leaf>"))
        handles.append(h)
    print(f"[NaN-trap][rank{rank}] attached {len(handles)} forward hooks", flush=True)
    return handles


class FirstBatchLoggerCallback(Callback):
    """Log the first training batch's ``input_ids`` stats (and optional
    vocab-OOB check), plus the per-step loss for the first ``n_steps``
    batches. Logging stops after that many batch-end calls.
    """

    def __init__(self, n_steps: int = 12, vocab_size: int | None = None):
        super().__init__()
        _validate_count("n_steps", n_steps)
        # Token IDs range over [0, vocab_size), so a vocabulary must be nonempty.
        if vocab_size is not None:
            _validate_count("vocab_size", vocab_size, minimum=1)
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self._logged_batch = False
        self._step_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        rank = os.environ.get("LOCAL_RANK", "0")
        if not self._logged_batch:
            ids = batch.get("input_ids") if isinstance(batch, dict) else None
            if isinstance(ids, torch.Tensor):
                msg = (
                    f"[first-batch][rank{rank}] input_ids shape={tuple(ids.shape)} "
                    f"dtype={ids.dtype}"
                )
                if ids.numel() == 0:
                    msg += f" bounds={MeasurementUnavailable('input_ids is empty').to_dict()}"
                elif ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    msg += f" bounds={MeasurementUnavailable('input_ids must have integer dtype').to_dict()}"
                else:
                    vmin = ids.min().item()
                    vmax = ids.max().item()
                    msg += f" min={vmin} max={vmax}"
                    if self.vocab_size is not None:
                        msg += f" vocab={self.vocab_size} oob={vmin < 0 or vmax >= self.vocab_size}"
                print(msg, flush=True)
            self._logged_batch = True

        if self._step_count < self.n_steps:
            loss_t = outputs.get("loss") if isinstance(outputs, dict) else outputs
            print(
                f"[step-trace][rank{rank}] step={trainer.global_step} "
                f"batch_idx={batch_idx} loss={_loss_summary(loss_t)}",
                flush=True,
            )
            self._step_count += 1
