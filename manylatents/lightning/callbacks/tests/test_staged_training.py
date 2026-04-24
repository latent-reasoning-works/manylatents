"""Tests for StagedTrainingCallback.

All tests run on CPU. Fixtures are intentionally tiny (Linear(10, 10) or a
2-layer synthetic transformer-like MLP) so each test completes in well under
5 seconds.
"""
from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from manylatents.lightning.callbacks.staged_training import (
    StagedTrainingCallback,
    _matches,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class TinyStudent(LightningModule):
    """Tiny LightningModule with named sub-modules for prefix-match tests.

    Structure:
      - ``embed``: Linear(10, 10)
      - ``layer.{0..3}``: Linear(10, 10) each
      - ``layer.10``, ``layer.11``, ``layer.12``: numeric siblings to test
        the overmatch guard.
      - ``head``: Linear(10, 1)
    """

    def __init__(self, lr: float = 1e-2) -> None:
        super().__init__()
        self.lr = lr
        self.embed = nn.Linear(10, 10)
        layer = nn.ModuleDict(
            {
                str(i): nn.Linear(10, 10)
                for i in (0, 1, 2, 3, 10, 11, 12)
            }
        )
        # Stash under attribute ``layer`` so named_parameters() produces
        # names like "layer.1.weight", "layer.10.weight", etc.
        self.layer = layer
        self.head = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for key in ("0", "1", "2", "3", "10", "11", "12"):
            h = self.layer[key](h)
        return self.head(h)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        pred = self(x)
        loss = ((pred - y) ** 2).mean()
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.lr)


def _make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 10)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _trainer(
    callback: StagedTrainingCallback,
    max_steps: int = 20,
) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )


# ---------------------------------------------------------------------------
# Pure-function prefix-matching tests
# ---------------------------------------------------------------------------
def test_matches_exact() -> None:
    assert _matches("layer.1", "layer.1")


def test_matches_dot_boundary() -> None:
    assert _matches("layer.1.weight", "layer.1")
    assert _matches("bert.encoder.layer.0.attention", "bert.encoder.layer.0")


def test_matches_rejects_numeric_sibling() -> None:
    assert not _matches("layer.10", "layer.1")
    assert not _matches("layer.11.weight", "layer.1")
    assert not _matches("layer.12", "layer.1")


def test_matches_rejects_unrelated_prefix() -> None:
    assert not _matches("head.weight", "layer.1")


# ---------------------------------------------------------------------------
# Init-time normalization
# ---------------------------------------------------------------------------
def test_trailing_dot_is_stripped() -> None:
    cb_a = StagedTrainingCallback(
        phase3_start_step=5, frozen_prefixes_phase2=["layer.1"]
    )
    cb_b = StagedTrainingCallback(
        phase3_start_step=5, frozen_prefixes_phase2=["layer.1."]
    )
    assert cb_a.frozen_prefixes_phase2 == cb_b.frozen_prefixes_phase2 == ["layer.1"]


def test_empty_prefix_is_rejected() -> None:
    with pytest.raises(ValueError):
        StagedTrainingCallback(phase3_start_step=5, frozen_prefixes_phase2=[""])
    with pytest.raises(ValueError):
        # A bare "." reduces to "" after rstrip.
        StagedTrainingCallback(phase3_start_step=5, frozen_prefixes_phase2=["."])


def test_negative_phase3_step_is_rejected() -> None:
    with pytest.raises(ValueError):
        StagedTrainingCallback(
            phase3_start_step=-1, frozen_prefixes_phase2=["layer.1"]
        )


# ---------------------------------------------------------------------------
# on_train_start behavior: freezing and optimizer rebuild
# ---------------------------------------------------------------------------
def test_on_train_start_freezes_matching_prefixes() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1", "layer.2"]
    )
    trainer = _trainer(cb, max_steps=1)
    trainer.fit(model, _make_loader())

    frozen_names = [
        name for name, p in model.named_parameters() if not p.requires_grad
    ]
    # Only layer.1.* and layer.2.* should be frozen.
    assert all(
        n.startswith("layer.1.") or n.startswith("layer.2.") for n in frozen_names
    )
    # And both blocks must be fully frozen (weight + bias).
    assert set(frozen_names) == {
        "layer.1.weight",
        "layer.1.bias",
        "layer.2.weight",
        "layer.2.bias",
    }


def test_on_train_start_keeps_non_matching_trainable() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=1)
    trainer.fit(model, _make_loader())

    for name in ("embed.weight", "embed.bias", "head.weight", "head.bias"):
        p = dict(model.named_parameters())[name]
        assert p.requires_grad, f"{name} should remain trainable"
    # Sanity-check a sibling that must not be overmatched.
    for name in ("layer.10.weight", "layer.11.weight", "layer.12.weight"):
        p = dict(model.named_parameters())[name]
        assert p.requires_grad, f"{name} should remain trainable"


def test_prefix_does_not_overmatch_numeric_siblings() -> None:
    """The core regression test: ``layer.1`` must NOT freeze ``layer.10``."""
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=1)
    trainer.fit(model, _make_loader())

    params = dict(model.named_parameters())
    assert not params["layer.1.weight"].requires_grad
    assert not params["layer.1.bias"].requires_grad
    for sib in ("10", "11", "12"):
        assert params[f"layer.{sib}.weight"].requires_grad
        assert params[f"layer.{sib}.bias"].requires_grad


def test_prefix_accepts_trailing_dot_or_not() -> None:
    model_a = TinyStudent()
    model_b = TinyStudent()
    # Use the same init so the freeze sets can be compared by name.
    model_b.load_state_dict(model_a.state_dict())

    cb_a = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1"]
    )
    cb_b = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1."]
    )
    _trainer(cb_a, max_steps=1).fit(model_a, _make_loader())
    _trainer(cb_b, max_steps=1).fit(model_b, _make_loader())

    frozen_a = {
        n for n, p in model_a.named_parameters() if not p.requires_grad
    }
    frozen_b = {
        n for n, p in model_b.named_parameters() if not p.requires_grad
    }
    assert frozen_a == frozen_b
    assert frozen_a == {"layer.1.weight", "layer.1.bias"}


def test_rebuild_optimizer_drops_frozen_params() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=100, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=1)
    trainer.fit(model, _make_loader())

    opt = trainer.optimizers[0]
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        if name.startswith("layer.1."):
            assert id(p) not in opt_param_ids, f"{name} should be dropped"
        else:
            assert id(p) in opt_param_ids, f"{name} should be retained"


# ---------------------------------------------------------------------------
# Phase3 boundary: unfreeze and rebuild preserving state
# ---------------------------------------------------------------------------
def test_unfreezes_at_phase3_boundary() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=3, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=6)
    trainer.fit(model, _make_loader(n=32))

    # After crossing step 3, all params must be trainable again.
    for name, p in model.named_parameters():
        assert p.requires_grad, f"{name} should be trainable post-phase3"
    assert cb._phase3_done is True


def test_rebuild_optimizer_reinstates_frozen_params() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=2, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=4)
    trainer.fit(model, _make_loader(n=32))

    opt = trainer.optimizers[0]
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        assert id(p) in opt_param_ids, f"{name} missing from phase3 optimizer"


def test_phase3_preserves_optimizer_state_for_surviving_params() -> None:
    """Params that trained through phase2 must retain their Adam moments
    after the phase3 rebuild.
    """
    model = TinyStudent(lr=1e-2)
    cb = StagedTrainingCallback(
        phase3_start_step=3, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=5)
    trainer.fit(model, _make_loader(n=40))

    opt = trainer.optimizers[0]
    # A param that was never frozen (e.g. ``embed.weight``) must have nonzero
    # Adam state after running through the phase3 boundary.
    embed_w = model.embed.weight
    assert embed_w in opt.state, "embed.weight should have Adam state"
    st = opt.state[embed_w]
    assert "exp_avg" in st
    assert "exp_avg_sq" in st
    # ``exp_avg`` should be nonzero because we took >= 1 step in phase2 on it.
    assert torch.any(st["exp_avg"] != 0), (
        "embed.weight's Adam exp_avg should have survived the phase3 rebuild"
    )
    assert torch.any(st["exp_avg_sq"] != 0)
    # ``step`` counter should reflect accumulated phase2 steps, not be reset.
    assert int(st["step"]) >= 1


def test_phase3_fresh_state_for_newly_unfrozen_params() -> None:
    """Previously-frozen params must start phase3 with empty Adam state,
    not leak any stale state from before they were frozen.
    """
    model = TinyStudent(lr=1e-2)
    cb = StagedTrainingCallback(
        phase3_start_step=3, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=4)
    # Run up to and including the phase3 boundary, but not past it so the
    # formerly-frozen param has not yet been stepped in phase3.
    trainer.fit(model, _make_loader(n=32))

    opt = trainer.optimizers[0]
    frozen_p = model.layer["1"].weight
    # Param was frozen throughout phase2 and has just been unfrozen at the
    # boundary. Immediately after the rebuild (and before any phase3 step has
    # touched it) its optimizer state entry must be empty. If trainer.fit
    # continued for additional steps past the boundary with nonzero grads,
    # Adam will have populated it; we assert the entry is either missing or
    # freshly populated (``step`` == 1 at most).
    st = opt.state.get(frozen_p, {})
    if st:
        assert int(st.get("step", 0)) <= 1, (
            "previously-frozen param should not have leaked phase2 Adam state"
        )


def test_phase3_transition_loss_bounded() -> None:
    """After the phase3 rebuild, loss must not spike more than 2x over the
    immediately-prior step. This is the empirical guard that the Adam state
    preservation actually works.
    """
    torch.manual_seed(0)

    class LossRecorder(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 1),
            )
            self.losses: List[float] = []

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = ((self(x) - y) ** 2).mean()
            self.losses.append(float(loss.detach()))
            return loss

        def configure_optimizers(self):
            trainable = [p for p in self.parameters() if p.requires_grad]
            return torch.optim.Adam(trainable, lr=1e-2)

    model = LossRecorder()
    # Freeze ``net.0`` for the first 10 steps, then unfreeze at step 10.
    cb = StagedTrainingCallback(
        phase3_start_step=10, frozen_prefixes_phase2=["net.0"]
    )
    trainer = _trainer(cb, max_steps=20)
    # Fixed batch → deterministic losses.
    torch.manual_seed(0)
    x = torch.randn(40, 10)
    y = torch.randn(40, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    trainer.fit(model, loader)

    assert len(model.losses) >= 11, (
        f"expected >= 11 logged losses, got {len(model.losses)}"
    )
    prev = model.losses[9]
    at_boundary = model.losses[10]
    assert at_boundary < prev * 2.0, (
        f"phase3 loss spike too large: losses[9]={prev:.4f}, "
        f"losses[10]={at_boundary:.4f}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_phase3_start_zero_skips_phase2() -> None:
    model = TinyStudent()
    cb = StagedTrainingCallback(
        phase3_start_step=0, frozen_prefixes_phase2=["layer.1"]
    )
    trainer = _trainer(cb, max_steps=2)
    trainer.fit(model, _make_loader())

    # No param should have been frozen: phase3_start_step=0 skips phase2.
    for name, p in model.named_parameters():
        assert p.requires_grad, f"{name} should remain trainable (phase2 skipped)"
    assert cb._phase3_done is True
