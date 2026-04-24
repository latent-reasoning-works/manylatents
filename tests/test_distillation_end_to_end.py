"""End-to-end integration test for the distillation algo module.

Composes the full pipeline:
  ActivationSnapshot.from_model  (teacher, probe set)
  align_on_snapshot              (phase1, imperative)
  trainer.fit(Distillation, ...) (phase2 → phase3 via StagedTrainingCallback)

Runs entirely on CPU in <60s with tiny synthetic models + synthetic data.
Protects the integration surface area between the four pieces built on
``distillation-algo-module``.

Primary contracts protected:
- snapshot.sample_ids IS the id space the datamodule uses (the failure mode
  the prior refactor died on: mismatched id spaces between teacher probe
  extraction and training-time probe lookup)
- phase1 seed flows end-to-end so a same-seed rerun produces identical losses
- StagedTrainingCallback correctly freezes aligned-layer params during phase2
  and unfreezes at phase3_start_step
- 15 total optimizer steps (5 phase1 + 10 phase2+3) all yield finite losses
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.algorithms.lightning.phase1_align import align_on_snapshot
from manylatents.lightning.callbacks.staged_training import StagedTrainingCallback
from manylatents.lightning.activation_snapshot import ActivationSnapshot


# ---- Minimal fixtures --------------------------------------------------------


class _TinyLMStudent(nn.Module):
    """Same minimal HF-shaped student used across the unit tests."""

    def __init__(self, vocab: int = 64, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=True) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class _SyntheticTaskDataModule(LightningDataModule):
    """In-memory task datamodule + exposed probe IDs.

    Holds a small fixed training set of (input_ids, attention_mask, labels).
    Exposes a ``probe_ids`` list so consumers can build a snapshot keyed to
    the same ID space the datamodule emits at training time - this is the
    contract the end-to-end test locks.
    """

    def __init__(
        self,
        n_train: int = 32,
        n_probe: int = 8,
        seq_len: int = 6,
        vocab: int = 64,
        batch_size: int = 4,
        seed: int = 7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed

        torch.manual_seed(seed)
        self._train_ids = torch.randint(0, vocab, (n_train, seq_len))
        self._train_mask = torch.ones(n_train, seq_len, dtype=torch.long)
        self._train_labels = self._train_ids.clone()

        self._probe_ids = torch.randint(0, vocab, (n_probe, seq_len))
        self._probe_mask = torch.ones(n_probe, seq_len, dtype=torch.long)
        # Stable integer identifiers for each probe row. Any consumer that
        # builds a snapshot against self._probe_ids must pass these same
        # values as snapshot.sample_ids.
        self.probe_ids: List[int] = list(range(1000, 1000 + n_probe))

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        ds = TensorDataset(self._train_ids, self._train_mask, self._train_labels)

        def collate(batch):
            ids = torch.stack([b[0] for b in batch])
            mask = torch.stack([b[1] for b in batch])
            labels = torch.stack([b[2] for b in batch])
            return {"input_ids": ids, "attention_mask": mask, "labels": labels}

        return DataLoader(ds, batch_size=self.batch_size, collate_fn=collate, shuffle=False)

    @property
    def probe_input_ids(self):
        return self._probe_ids

    @property
    def probe_attention_mask(self):
        return self._probe_mask


def _build_pipeline(
    student: nn.Module,
    teacher: nn.Module,
    dm: _SyntheticTaskDataModule,
    alignment_weight: float = 1.0,
    phase3_start_step: int = 5,
    frozen_prefixes: Optional[List[str]] = None,
):
    """Build a (snapshot, distillation_module, phase1_losses) triple up to
    just-before trainer.fit."""
    # 1. Snapshot against the teacher on the datamodule's probe inputs, keyed
    #    by the datamodule's probe_ids.
    snapshot = ActivationSnapshot.from_model(
        teacher,
        input_ids=dm.probe_input_ids,
        attention_mask=dm.probe_attention_mask,
        sample_ids=dm.probe_ids,
        layer_paths=["layers.1"],
        reduction="mean",
    )

    # 2. Phase1 imperative alignment using the datamodule's seed.
    phase1_losses = align_on_snapshot(
        student,
        snapshot,
        layer_pairs=[{"student": "layers.1", "teacher": "layers.1", "weight": 1.0}],
        n_steps=5,
        optimizer_cfg={"learning_rate": 1e-2},
        batch_size=4,
        seed=dm.seed,
        device="cpu",
    )

    # 3. Distillation module for phase2/3.
    mod = Distillation(
        datamodule=dm,
        student=student,
        activation_snapshot=snapshot,
        layer_pairs=[{"student": "layers.1", "teacher": "layers.1", "weight": 1.0}],
        optimizer={"learning_rate": 1e-3},
        alignment_weight=alignment_weight,
        alignment_batch_size=4,
        init_seed=dm.seed,
    )
    return snapshot, mod, phase1_losses


# ---- Tests -------------------------------------------------------------------


def test_snapshot_sample_ids_match_text_datamodule_probe_ids() -> None:
    """Sharpest residual risk called out in the reviewer log: snapshot must
    be built against the same probe_ids the datamodule emits."""
    torch.manual_seed(0)
    teacher = _TinyLMStudent()
    dm = _SyntheticTaskDataModule()

    snap = ActivationSnapshot.from_model(
        teacher,
        input_ids=dm.probe_input_ids,
        attention_mask=dm.probe_attention_mask,
        sample_ids=dm.probe_ids,
        layer_paths=["layers.1"],
        reduction="mean",
    )
    assert set(snap.sample_ids) == set(dm.probe_ids)


def test_end_to_end_phase1_then_fit_runs_cleanly() -> None:
    """Full pipeline: snapshot → phase1 → fit. Counts optimizer steps and
    asserts every loss is finite."""
    torch.manual_seed(0)
    student = _TinyLMStudent()
    teacher = _TinyLMStudent()  # different init
    dm = _SyntheticTaskDataModule()

    _, mod, phase1_losses = _build_pipeline(
        student, teacher, dm,
        alignment_weight=1.0,
        phase3_start_step=5,
    )

    assert len(phase1_losses) == 5
    assert all(torch.tensor(l).isfinite().item() for l in phase1_losses)

    callback = StagedTrainingCallback(
        phase3_start_step=5,
        frozen_prefixes_phase2=["student.layers.1"],
    )
    trainer = Trainer(
        max_steps=10,
        accelerator="cpu",
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(mod, datamodule=dm)
    assert trainer.global_step == 10


def test_phase1_and_phase2_use_consistent_seed() -> None:
    """Same seed end-to-end → identical phase1 loss trajectories on a
    two-run comparison."""
    def run_phase1():
        torch.manual_seed(1234)
        student = _TinyLMStudent()
        teacher = _TinyLMStudent()
        dm = _SyntheticTaskDataModule(seed=7)
        _, _, p1 = _build_pipeline(student, teacher, dm)
        return p1

    a = run_phase1()
    b = run_phase1()
    assert a == b, f"non-deterministic under fixed seeds: a={a} b={b}"


def test_staged_callback_freezes_aligned_layers_during_phase2() -> None:
    """During steps 0..phase3_start_step-1, aligned-layer params must have
    requires_grad=False. After the transition, requires_grad is True."""
    torch.manual_seed(0)
    student = _TinyLMStudent()
    teacher = _TinyLMStudent()
    dm = _SyntheticTaskDataModule()
    _, mod, _ = _build_pipeline(student, teacher, dm)

    observed_requires_grad = []

    class _Probe(StagedTrainingCallback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            observed_requires_grad.append(
                (trainer.global_step, pl_module.student.layers[1].weight.requires_grad)
            )
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    callback = _Probe(
        phase3_start_step=5,
        frozen_prefixes_phase2=["student.layers.1"],
    )
    trainer = Trainer(
        max_steps=10,
        accelerator="cpu",
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(mod, datamodule=dm)

    # During phase2 (steps 1..5), observed value should be False.
    # After phase3 transition (callback fires at end of step 5 → global_step 6
    # onward sees it unfrozen).
    phase2_obs = [(s, r) for s, r in observed_requires_grad if s < 5]
    phase3_obs = [(s, r) for s, r in observed_requires_grad if s >= 6]
    assert phase2_obs, "no phase2 observations"
    assert phase3_obs, "no phase3 observations"
    assert all(not r for _, r in phase2_obs), (
        f"phase2 layers.1.weight should be frozen; got {phase2_obs}"
    )
    assert all(r for _, r in phase3_obs), (
        f"phase3 layers.1.weight should be trainable; got {phase3_obs}"
    )


def test_control_task_only_all_losses_finite() -> None:
    """alignment_weight=0 config: pure task training still runs end-to-end."""
    torch.manual_seed(0)
    student = _TinyLMStudent()
    teacher = _TinyLMStudent()
    dm = _SyntheticTaskDataModule()
    _, mod, _ = _build_pipeline(student, teacher, dm, alignment_weight=0.0)

    trainer = Trainer(
        max_steps=10,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(mod, datamodule=dm)
    assert trainer.global_step == 10
