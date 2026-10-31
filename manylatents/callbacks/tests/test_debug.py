"""Unit tests for callbacks.debug — NaN detector + FirstBatchLoggerCallback."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
import torch
from torch import nn

from manylatents.callbacks.debug import (
    FirstBatchLoggerCallback,
    attach_nan_detector,
)


class _NaNLeaf(nn.Module):
    """Leaf module whose forward emits NaN deterministically."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = x.clone()
        out[..., 0] = float("nan")
        return out


class _InfLeaf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = x.clone()
        out[..., 0] = float("inf")
        return out


class _NaNEmitter(nn.Module):
    """Wrapper around a NaN-emitting leaf, to test that the hook fires on
    the leaf (not the parent)."""

    def __init__(self) -> None:
        super().__init__()
        self.leaf = _NaNLeaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.leaf(x)


class _CleanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.lin(x)


class _FakeTrainer:
    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step


def _run_with_capture(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def test_nan_detector_reports_nan() -> None:
    model = _NaNEmitter()
    handles = attach_nan_detector(model, max_reports=10)
    out = _run_with_capture(lambda: model(torch.zeros(2, 4)))
    for h in handles:
        h.remove()
    assert "FIRST_BAD" in out
    assert "NaN=" in out
    assert "module='leaf'" in out


def test_nan_detector_silent_on_clean_model() -> None:
    model = _CleanModel()
    handles = attach_nan_detector(model, max_reports=10)
    out = _run_with_capture(lambda: model(torch.zeros(2, 4)))
    for h in handles:
        h.remove()
    assert "FIRST_BAD" not in out


def test_nan_detector_max_reports_cap() -> None:
    model = _NaNEmitter()
    handles = attach_nan_detector(model, max_reports=2)

    def runner() -> None:
        for _ in range(5):
            model(torch.zeros(2, 4))

    out = _run_with_capture(runner)
    for h in handles:
        h.remove()
    assert out.count("FIRST_BAD") == 2
    assert "reached max_reports=2" in out


def test_nan_detector_detects_inf() -> None:
    class _InfEmitter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.leaf = _InfLeaf()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.leaf(x)

    model = _InfEmitter()
    handles = attach_nan_detector(model, max_reports=10)
    out = _run_with_capture(lambda: model(torch.zeros(2, 4)))
    for h in handles:
        h.remove()
    assert "FIRST_BAD" in out
    assert "Inf=" in out


def test_first_batch_logger_logs_input_ids_and_oob() -> None:
    cb = FirstBatchLoggerCallback(n_steps=0, vocab_size=100)
    batch = {"input_ids": torch.tensor([[5, 50, 99], [10, 20, 30]])}
    out = _run_with_capture(
        lambda: cb.on_train_batch_end(_FakeTrainer(), None, {"loss": 0.0}, batch, 0)
    )
    assert "[first-batch]" in out
    assert "min=5" in out
    assert "max=99" in out
    assert "vocab=100" in out
    assert "oob=False" in out


def test_first_batch_logger_oob_true_when_id_at_vocab() -> None:
    cb = FirstBatchLoggerCallback(n_steps=0, vocab_size=50)
    batch = {"input_ids": torch.tensor([[1, 2, 50]])}
    out = _run_with_capture(
        lambda: cb.on_train_batch_end(_FakeTrainer(), None, {"loss": 0.0}, batch, 0)
    )
    assert "oob=True" in out


def test_first_batch_logger_step_trace_unplugs_after_n_steps() -> None:
    cb = FirstBatchLoggerCallback(n_steps=2, vocab_size=None)
    batch = {"input_ids": torch.tensor([[1, 2]])}

    def runner() -> None:
        for step in range(5):
            cb.on_train_batch_end(_FakeTrainer(global_step=step), None, {"loss": 0.5}, batch, step)

    out = _run_with_capture(runner)
    assert out.count("[step-trace]") == 2


def test_first_batch_logger_handles_non_dict_batch() -> None:
    cb = FirstBatchLoggerCallback(n_steps=1, vocab_size=None)
    out = _run_with_capture(
        lambda: cb.on_train_batch_end(_FakeTrainer(), None, {"loss": 1.0}, torch.zeros(2, 4), 0)
    )
    assert "[first-batch]" not in out
    assert "[step-trace]" in out


def test_first_batch_logger_handles_bare_tensor_loss() -> None:
    cb = FirstBatchLoggerCallback(n_steps=1, vocab_size=None)
    bare_loss = torch.tensor(0.42)
    out = _run_with_capture(
        lambda: cb.on_train_batch_end(_FakeTrainer(), None, bare_loss, {"input_ids": torch.tensor([[1, 2]])}, 0)
    )
    assert "loss=4.200000e-01" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
