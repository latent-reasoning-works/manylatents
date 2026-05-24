"""Unit tests for snapshot_projection.phate_procrustes."""
from __future__ import annotations

import pytest
import torch

from manylatents.algorithms.snapshot_projection import (
    pca_reduce,
    procrustes_align,
    project_to_student_dim,
)
from manylatents.lightning.activation_snapshot import ActivationSnapshot


def _make_snapshot(n: int = 32, teacher_dim: int = 16, n_layers: int = 1, *, dtype=torch.float32, device="cpu") -> ActivationSnapshot:
    torch.manual_seed(0)
    return ActivationSnapshot(
        input_ids=torch.zeros(n, 8, dtype=torch.int64),
        attention_mask=torch.ones(n, 8, dtype=torch.int64),
        sample_ids=list(range(n)),
        activations={
            f"layer.{i}": torch.randn(n, teacher_dim, dtype=dtype, device=device)
            for i in range(n_layers)
        },
        reduction="mean",
    )


def test_pca_reduce_shape_and_finite() -> None:
    x = torch.randn(50, 32)
    out = pca_reduce(x, n_components=8)
    assert out.shape == (50, 8)
    assert torch.isfinite(out).all()


def test_pca_reduce_caps_at_rank() -> None:
    x = torch.randn(20, 5)
    out = pca_reduce(x, n_components=20)
    assert out.shape == (20, 5)


def test_pca_reduce_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        pca_reduce(torch.zeros(3), n_components=2)


def test_procrustes_same_dim_returns_correct_shape() -> None:
    src = torch.randn(20, 4)
    tgt = torch.randn(20, 4)
    out = procrustes_align(src, tgt)
    assert out.shape == (20, 4)
    assert torch.isfinite(out).all()


def test_procrustes_lower_dim_target_is_pca_reduced() -> None:
    src = torch.randn(20, 4)
    tgt = torch.randn(20, 64)
    out = procrustes_align(src, tgt)
    assert out.shape == (20, 4)
    assert torch.isfinite(out).all()


def test_procrustes_rejects_n_below_2() -> None:
    src = torch.randn(1, 3)
    tgt = torch.randn(1, 3)
    with pytest.raises(ValueError):
        procrustes_align(src, tgt)


def test_project_shape_and_finite() -> None:
    snap = _make_snapshot(n=24, teacher_dim=16, n_layers=1)
    out = project_to_student_dim(
        snap,
        student_hidden_dim=8,
        knn=5,
        t="auto",
        decay=40,
        gamma=1.0,
        random_state=0,
    )
    proj = next(iter(out.activations.values()))
    assert proj.shape == (24, 8)
    assert torch.isfinite(proj).all()


def test_project_preserves_metadata() -> None:
    snap = _make_snapshot(n=24, teacher_dim=16, n_layers=2)
    out = project_to_student_dim(
        snap, student_hidden_dim=8, random_state=0,
    )
    assert torch.equal(out.input_ids, snap.input_ids)
    assert torch.equal(out.attention_mask, snap.attention_mask)
    assert out.sample_ids == snap.sample_ids
    assert out.reduction == snap.reduction
    assert set(out.activations.keys()) == set(snap.activations.keys())


def test_project_no_procrustes_path() -> None:
    snap = _make_snapshot(n=24, teacher_dim=16)
    out = project_to_student_dim(
        snap,
        student_hidden_dim=8,
        random_state=0,
        procrustes_align_to_teacher=False,
    )
    proj = next(iter(out.activations.values()))
    assert proj.shape == (24, 8)
    assert torch.isfinite(proj).all()


def test_project_dtype_round_trip() -> None:
    snap = _make_snapshot(n=24, teacher_dim=16, dtype=torch.float64)
    out = project_to_student_dim(
        snap, student_hidden_dim=8, random_state=0,
    )
    proj = next(iter(out.activations.values()))
    assert proj.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
