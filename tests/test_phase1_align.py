"""Unit tests for align_on_snapshot phase1 helper."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manylatents.algorithms.lightning.phase1_align import align_on_snapshot
from manylatents.lightning.activation_snapshot import ActivationSnapshot


class _TinyStudent(nn.Module):
    """Minimal transformer-shaped student for phase1 tests. Same shape as
    the _TinyBertLike in test_activation_snapshot but without a vocab head
    (phase1 only needs hidden activations, not token logits)."""

    def __init__(self, vocab: int = 100, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


def _build_snapshot(n: int = 16, hidden: int = 8, seed: int = 0) -> ActivationSnapshot:
    torch.manual_seed(seed)
    # Build a snapshot from a DIFFERENT random init so MSE is non-trivial.
    teacher = _TinyStudent(hidden=hidden)
    input_ids = torch.randint(0, 100, (n, 6), dtype=torch.long)
    attention_mask = torch.ones(n, 6, dtype=torch.long)
    return ActivationSnapshot.from_model(
        teacher,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sample_ids=list(range(n)),
        layer_paths=["layers.1"],
        reduction="mean",
    )


def _default_pairs():
    return [{"student": "layers.1", "teacher": "layers.1", "weight": 1.0}]


def test_align_reduces_mse_monotone_window() -> None:
    """20 steps on a tiny fixture: mean of last 5 losses < mean of first 5."""
    torch.manual_seed(42)
    student = _TinyStudent()
    snap = _build_snapshot(seed=7)

    losses = align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=20,
        optimizer_cfg={"learning_rate": 1e-2},
        batch_size=8, seed=123, device="cpu",
    )

    assert len(losses) == 20
    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early, f"expected decrease; early={early:.4f} late={late:.4f}"


def test_align_deterministic_under_seed() -> None:
    """Same seed, same student init, same snapshot → identical trajectories."""
    def run():
        torch.manual_seed(999)
        student = _TinyStudent()
        snap = _build_snapshot(seed=7)
        return align_on_snapshot(
            student, snap, _default_pairs(),
            n_steps=5,
            optimizer_cfg={"learning_rate": 1e-2},
            batch_size=4, seed=42, device="cpu",
        )

    a = run()
    b = run()
    assert a == b, f"non-deterministic: a={a} b={b}"


def test_align_n_steps_zero_returns_empty() -> None:
    student = _TinyStudent()
    snap = _build_snapshot()
    losses = align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=0,
        optimizer_cfg={"learning_rate": 1e-3},
        device="cpu",
    )
    assert losses == []


def test_align_restores_student_hooks_removed() -> None:
    """After return, no forward hooks should remain on any submodule."""
    student = _TinyStudent()
    snap = _build_snapshot()
    align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=3,
        optimizer_cfg={"learning_rate": 1e-3},
        device="cpu",
    )
    for m in student.modules():
        assert len(m._forward_hooks) == 0, (
            f"lingering hook on {type(m).__name__}"
        )


def test_align_respects_layer_pair_weights() -> None:
    """Doubling pair weights doubles the loss values."""
    torch.manual_seed(0)
    student_a = _TinyStudent()
    torch.manual_seed(0)
    student_b = _TinyStudent()
    snap = _build_snapshot(seed=7)

    losses_w1 = align_on_snapshot(
        student_a, snap,
        [{"student": "layers.1", "teacher": "layers.1", "weight": 1.0}],
        n_steps=1,
        optimizer_cfg={"learning_rate": 1e-10},  # tiny LR -> weights ~ unchanged
        batch_size=8, seed=42, device="cpu",
    )
    losses_w2 = align_on_snapshot(
        student_b, snap,
        [{"student": "layers.1", "teacher": "layers.1", "weight": 2.0}],
        n_steps=1,
        optimizer_cfg={"learning_rate": 1e-10},
        batch_size=8, seed=42, device="cpu",
    )
    assert losses_w2[0] == pytest.approx(2 * losses_w1[0], rel=1e-3)


def test_align_raises_on_device_mismatch() -> None:
    """Snapshot on one device, align_on_snapshot asked for another → raise."""
    student = _TinyStudent()
    snap = _build_snapshot()  # on CPU
    with pytest.raises(ValueError, match=r"device=meta.*other devices"):
        align_on_snapshot(
            student, snap, _default_pairs(),
            n_steps=1,
            optimizer_cfg={"learning_rate": 1e-3},
            device="meta",
        )


def test_align_raises_on_unresolvable_student_path() -> None:
    student = _TinyStudent()
    snap = _build_snapshot()
    with pytest.raises(ValueError, match=r"does not resolve"):
        align_on_snapshot(
            student, snap,
            [{"student": "nonexistent.path", "teacher": "layers.1", "weight": 1.0}],
            n_steps=1,
            optimizer_cfg={"learning_rate": 1e-3},
            device="cpu",
        )


def test_align_uses_fresh_optimizer() -> None:
    """Pre-existing optimizer attr on the student must not be touched."""
    student = _TinyStudent()
    pre_existing_marker = torch.optim.SGD(student.parameters(), lr=0.0)
    student.optimizer = pre_existing_marker  # type: ignore[attr-defined]
    snap = _build_snapshot()

    align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=2,
        optimizer_cfg={"learning_rate": 1e-3},
        device="cpu",
    )
    assert student.optimizer is pre_existing_marker  # type: ignore[attr-defined]


def test_align_updates_only_requires_grad_true() -> None:
    """Frozen params should not receive gradient updates during phase1."""
    student = _TinyStudent()
    for p in student.embed.parameters():
        p.requires_grad = False
    embed_before = student.embed.weight.detach().clone()

    snap = _build_snapshot()
    align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=5,
        optimizer_cfg={"learning_rate": 1e-1},
        device="cpu",
    )
    assert torch.equal(student.embed.weight.detach(), embed_before), (
        "frozen embed must not change during phase1"
    )


def test_align_losses_are_finite() -> None:
    student = _TinyStudent()
    snap = _build_snapshot()
    losses = align_on_snapshot(
        student, snap, _default_pairs(),
        n_steps=10,
        optimizer_cfg={"learning_rate": 1e-3},
        batch_size=8, seed=1, device="cpu",
    )
    assert all(float(l) == float(l) for l in losses), "NaN loss"  # NaN check
    assert all(abs(l) < 1e6 for l in losses), f"diverged: {losses}"


def test_align_seed_variation_produces_different_losses() -> None:
    """Different seeds should produce different per-step losses (else the
    sampler is deterministic in the wrong dimension)."""
    def run(seed):
        torch.manual_seed(0)
        student = _TinyStudent()
        snap = _build_snapshot(seed=7)
        return align_on_snapshot(
            student, snap, _default_pairs(),
            n_steps=3,
            optimizer_cfg={"learning_rate": 1e-6},  # so loss ≈ initial MSE shape
            batch_size=4, seed=seed, device="cpu",
        )

    a = run(11)
    b = run(12)
    assert a != b, f"different seeds produced identical losses: {a}"
