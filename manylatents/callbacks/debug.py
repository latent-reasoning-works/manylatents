"""Opt-in debug callbacks: NaN/Inf detector and first-batch logger.

Both surfaces are imported by callers that gate them on env vars (e.g.
``DEBUG_NAN_HOOKS=1``). Nothing here is wired into manylatents core flows.
"""
from __future__ import annotations

import os
from typing import Any, List

import torch
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.utils.hooks import RemovableHandle


def attach_nan_detector(model: nn.Module, max_reports: int = 10) -> List[RemovableHandle]:
    """Register forward hooks on every leaf module that print the first
    ``max_reports`` NaN/Inf occurrences in outputs and then go silent.

    Returns the list of handles so callers can remove them later.
    """
    rank = os.environ.get("LOCAL_RANK", "0")
    state = {"reports": 0, "max": int(max_reports)}
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
                if t.numel() > 0 and bool(finite_mask.any().item()):
                    amax = float(t[finite_mask].abs().max().item())
                else:
                    amax = float("nan")
                n_nan = int(nan_mask.sum().item())
                n_inf = int(inf_mask.sum().item())
                print(
                    f"[NaN-trap][rank{rank}] FIRST_BAD module={tname!r} "
                    f"shape={tuple(t.shape)} dtype={t.dtype} "
                    f"NaN={n_nan} Inf={n_inf} "
                    f"finite_abs_max={amax:.4e}",
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
    optimizer steps. Then unplugs.
    """

    def __init__(self, n_steps: int = 12, vocab_size: int | None = None):
        super().__init__()
        self.n_steps = int(n_steps)
        self.vocab_size = vocab_size
        self._logged_batch = False
        self._step_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        rank = os.environ.get("LOCAL_RANK", "0")
        if not self._logged_batch:
            ids = batch.get("input_ids") if isinstance(batch, dict) else None
            if isinstance(ids, torch.Tensor):
                vmin = int(ids.min().item())
                vmax = int(ids.max().item())
                msg = (
                    f"[first-batch][rank{rank}] input_ids shape={tuple(ids.shape)} "
                    f"dtype={ids.dtype} min={vmin} max={vmax}"
                )
                if self.vocab_size is not None:
                    msg += f" vocab={self.vocab_size} oob={vmax >= self.vocab_size}"
                print(msg, flush=True)
            self._logged_batch = True

        if self._step_count < self.n_steps:
            loss_t = outputs.get("loss") if isinstance(outputs, dict) else outputs
            try:
                loss_v = float(loss_t)
            except Exception:
                loss_v = float("nan")
            print(
                f"[step-trace][rank{rank}] step={trainer.global_step} "
                f"batch_idx={batch_idx} loss={loss_v:.6e}",
                flush=True,
            )
            self._step_count += 1
