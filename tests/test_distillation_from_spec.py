"""Unit tests for Distillation.from_spec(dict)."""
from __future__ import annotations

import copy
from typing import Optional

import pytest
import torch
import torch.nn as nn

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.lightning.activation_snapshot import ActivationSnapshot


class _TinyStudent(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8, n_layers: int = 1) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layer = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, vocab)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.embed(input_ids)
        for blk in self.layer:
            x = blk(x)
        return {"loss": x.sum() * 0.0, "logits": self.head(x)}


def _snapshot(n: int = 4, hidden: int = 8) -> ActivationSnapshot:
    return ActivationSnapshot(
        input_ids=torch.zeros(n, 4, dtype=torch.int64),
        attention_mask=torch.ones(n, 4, dtype=torch.int64),
        sample_ids=list(range(n)),
        activations={"layer.0": torch.zeros(n, hidden)},
        reduction="mean",
    )


def _spec_minimal(*, staged: bool = False) -> dict:
    spec: dict = {
        "reproducibility": {
            "optimizer": {
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
            "alignment": {"batch_size": 4},
            "seeds": {"global_seed": 7},
            "training": {"max_steps": 1000},
            "lr_scheduler": {"warmup_steps": 50},
        }
    }
    if staged:
        spec["reproducibility"]["training"]["staged_training"] = {
            "enabled": True,
            "phase2": {"max_steps": 600},
            "phase3": {"max_steps": 400},
        }
    return spec


def _build(spec: dict, **overrides):
    return Distillation.from_spec(
        spec,
        student=_TinyStudent(),
        activation_snapshot=_snapshot(),
        layer_pairs=[],
        datamodule=None,
        **overrides,
    )


def test_from_spec_returns_distillation() -> None:
    mod = _build(_spec_minimal())
    assert isinstance(mod, Distillation)


def test_from_spec_optimizer_field_mapping() -> None:
    mod = _build(_spec_minimal())
    opt = mod.optimizer_cfg
    assert opt["learning_rate"] == 1e-4
    assert opt["weight_decay"] == 0.01
    assert opt["betas"] == (0.9, 0.95)
    assert opt["eps"] == 1e-8


def test_from_spec_staged_total_steps() -> None:
    mod = _build(_spec_minimal(staged=True))
    assert mod.lr_scheduler_cfg is not None
    assert mod.lr_scheduler_cfg["total_steps"] == 1000  # 600 + 400
    assert mod.lr_scheduler_cfg["warmup_steps"] == 50


def test_from_spec_non_staged_total_steps() -> None:
    mod = _build(_spec_minimal(staged=False))
    assert mod.lr_scheduler_cfg is not None
    assert mod.lr_scheduler_cfg["total_steps"] == 1000


def test_from_spec_omits_scheduler_when_absent() -> None:
    spec = _spec_minimal()
    del spec["reproducibility"]["lr_scheduler"]
    mod = _build(spec)
    assert mod.lr_scheduler_cfg is None


def test_from_spec_alignment_weight_pass_through() -> None:
    mod = _build(_spec_minimal(), alignment_weight=0.0)
    assert mod.alignment_weight == 0.0


def test_from_spec_alignment_weight_default_zero() -> None:
    mod = _build(_spec_minimal())
    assert mod.alignment_weight == 0.0


def test_from_spec_pulls_global_seed() -> None:
    mod = _build(_spec_minimal())
    assert mod.init_seed == 7


def test_from_spec_missing_required_key_raises() -> None:
    spec = _spec_minimal()
    del spec["reproducibility"]["optimizer"]
    with pytest.raises(KeyError):
        _build(spec)


def test_from_spec_alignment_batch_size_mapping() -> None:
    spec = _spec_minimal()
    spec["reproducibility"]["alignment"]["batch_size"] = 32
    mod = _build(spec)
    assert mod.alignment_batch_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
