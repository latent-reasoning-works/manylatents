"""Distillation LightningModule.

Student training with an optional alignment regularizer sampled from a frozen
:class:`~manylatents.lightning.activation_snapshot.ActivationSnapshot`. Mirrors
the shape of :class:`manylatents.algorithms.lightning.reconstruction.Reconstruction`
and :class:`manylatents.lightning.hf_trainer.HFTrainerModule`: one ``__init__``
that captures configuration, one ``configure_model`` / ``setup`` that binds the
student, one ``training_step`` that emits the combined loss.

Phase1 alignment (pre-training against the snapshot's activations as targets)
is NOT part of this module - it lives in
``manylatents.algorithms.lightning.phase1_align.align_on_snapshot`` and runs
imperatively before ``trainer.fit``. The callback that handles freeze/unfreeze
between phase2 and phase3 lives in
``manylatents.lightning.callbacks.staged_training.StagedTrainingCallback``.

See ``/network/scratch/c/cesar.valdez/distillation-algo-module-plan.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor

from manylatents.lightning.activation_snapshot import ActivationSnapshot
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec, resolve_layer

__all__ = ["Distillation"]


def _sanitize_buffer_name(layer_path: str) -> str:
    """Translate a dotted layer path to a valid Python identifier for register_buffer."""
    return "_target__" + (
        layer_path.replace(".", "_").replace("[", "_").replace("]", "_").replace("-", "m")
    )


def _collect_norm_param_ids(student: nn.Module) -> set:
    """Return ids of all parameters living inside any norm module.

    Matches LayerNorm and any other torch module whose class name ends with
    ``Norm`` (covers RMSNorm, GroupNorm, BatchNorm, and third-party variants).
    Module-type matching is more robust than string-matching on param names,
    which fails for modules named ``final_ln`` / ``attn_norm`` / etc.
    """
    norm_param_ids = set()
    for module in student.modules():
        if isinstance(module, nn.LayerNorm) or type(module).__name__.endswith("Norm"):
            for p in module.parameters(recurse=False):
                norm_param_ids.add(id(p))
    return norm_param_ids


def _split_param_groups(
    student: nn.Module,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Two AdamW param groups: weight-decayed for normal weights,
    weight_decay=0 for biases and norm weights (standard HF recipe).

    Only params with ``requires_grad=True`` are included. This is what makes
    :class:`~manylatents.lightning.callbacks.staged_training.StagedTrainingCallback`'s
    freeze semantic actually freeze - dropping a param from the optimizer
    requires us to not include it here in the first place.
    """
    norm_ids = _collect_norm_param_ids(student)
    decay, no_decay = [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or id(p) in norm_ids:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Distillation(LightningModule):
    """Student training with optional frozen-snapshot alignment regularizer.

    Args:
        datamodule: Lightning datamodule providing ``train_dataloader`` (and
            optionally ``val_dataloader``). Each batch must be ``dict``-shaped
            with keys the student's ``forward`` accepts - typically
            ``input_ids``, ``attention_mask``, and ``labels``.
        student: pre-instantiated ``nn.Module`` for the student. The Hydra
            config layer (see ``configs/algorithms/lightning/distillation/``)
            wraps this with a ``_target_`` pointing at a builder function.
        activation_snapshot: frozen reference snapshot used for the alignment
            regularizer. Tensors are registered as buffers so ``.to(device)``
            and checkpointing Just Work.
        layer_pairs: list of
            ``{"student": <student_layer_path>, "teacher": <snapshot_key>, "weight": <float>}``.
            Only used when ``alignment_weight > 0``.
        optimizer: AdamW config. Recognised keys: ``learning_rate`` (or ``lr``),
            ``weight_decay``, ``betas``, ``eps``. Anything else ignored.
        alignment_weight: coefficient on the alignment MSE term. ``0.0`` means
            pure task training; the module never runs the probe forward pass.
        alignment_batch_size: number of probe samples drawn per
            ``training_step`` when computing the alignment term.
        init_seed: seed used before optimizer construction. Student weights are
            the caller's responsibility - init the module deterministically
            upstream if you need reproducible runs.
        lr_scheduler: optional scheduler config. Keys: ``warmup_steps``,
            ``total_steps``. Uses a linear warmup+decay schedule.
    """

    def __init__(
        self,
        datamodule,
        student: nn.Module,
        activation_snapshot: ActivationSnapshot,
        layer_pairs: List[Mapping[str, Any]],
        optimizer: Mapping[str, Any],
        *,
        alignment_weight: float = 0.0,
        alignment_batch_size: int = 16,
        init_seed: int = 42,
        lr_scheduler: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()

        if alignment_weight > 0 and len(layer_pairs) == 0:
            raise ValueError(
                "layer_pairs must be non-empty when alignment_weight > 0"
            )

        # Reduction is a property of the snapshot, NOT a duplicate here.
        # _alignment_loss reads self.snapshot.reduction directly.
        self._snapshot = activation_snapshot

        # Validate layer_pairs against the snapshot's keys up front. Failing
        # early avoids a confusing KeyError deep inside _alignment_loss.
        if alignment_weight > 0:
            snapshot_layers = set(activation_snapshot.activations.keys())
            for pair in layer_pairs:
                if pair["teacher"] not in snapshot_layers:
                    raise ValueError(
                        f"layer_pairs teacher={pair['teacher']!r} is not in "
                        f"snapshot.activations (keys: {sorted(snapshot_layers)})"
                    )

        self.datamodule = datamodule
        self.student = student
        self.layer_pairs = [dict(p) for p in layer_pairs]
        self.optimizer_cfg = dict(optimizer)
        self.lr_scheduler_cfg = dict(lr_scheduler) if lr_scheduler is not None else None
        self.alignment_weight = float(alignment_weight)
        self.alignment_batch_size = int(alignment_batch_size)
        self.init_seed = int(init_seed)

        # Register snapshot tensors as buffers so .to(device) moves them and
        # Lightning saves them in checkpoints. Biased copy (.clone()) so
        # snapshot lifecycle is independent of the module's.
        self.register_buffer("_probe_input_ids", activation_snapshot.input_ids.clone())
        self.register_buffer(
            "_probe_attention_mask", activation_snapshot.attention_mask.clone()
        )
        self._layer_buffer_names: Dict[str, str] = {}
        for layer_path, acts in activation_snapshot.activations.items():
            buf_name = _sanitize_buffer_name(layer_path)
            self.register_buffer(buf_name, acts.clone())
            self._layer_buffer_names[layer_path] = buf_name

        # Validate student layer paths resolve (only if we'll actually use them).
        if alignment_weight > 0:
            for pair in layer_pairs:
                try:
                    resolve_layer(student, pair["student"])
                except (AttributeError, IndexError) as exc:
                    raise ValueError(
                        f"layer_pairs student={pair['student']!r} does not "
                        f"resolve on the provided student module: {exc}"
                    ) from exc

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, Any],
        *,
        student: nn.Module,
        activation_snapshot: ActivationSnapshot,
        layer_pairs: List[Mapping[str, Any]],
        datamodule: Any,
        alignment_weight: float = 0.0,
    ) -> "Distillation":
        """Build a Distillation from a run-spec dict.

        Reads ``spec["reproducibility"]["{optimizer, alignment, seeds, training, lr_scheduler}"]``
        and maps them onto ``__init__`` kwargs. ``total_steps`` for the LR
        schedule is the sum of phase2+phase3 ``max_steps`` when
        ``training.staged_training.enabled`` is True, otherwise
        ``training.max_steps``.

        Required spec keys: ``reproducibility.optimizer``,
        ``reproducibility.alignment.batch_size``,
        ``reproducibility.seeds.global_seed``, ``reproducibility.training``.
        ``reproducibility.lr_scheduler`` is optional. Missing required keys
        raise KeyError eagerly.

        HF model and datamodule construction stay in the caller; this method
        only wires already-built objects into the Distillation kwargs.
        """
        repro = spec["reproducibility"]

        opt = repro["optimizer"]
        optimizer = {
            "learning_rate": float(opt["learning_rate"]),
            "weight_decay": float(opt["weight_decay"]),
            "betas": tuple(opt["betas"]),
            "eps": float(opt["eps"]),
        }

        training = repro["training"]
        staged = training.get("staged_training") or {}
        if bool(staged.get("enabled", False)):
            total_steps = int(staged["phase2"]["max_steps"]) + int(
                staged["phase3"]["max_steps"]
            )
        else:
            total_steps = int(training["max_steps"])

        lr_scheduler: Optional[Dict[str, Any]] = None
        if "lr_scheduler" in repro:
            lr_scheduler = {
                "warmup_steps": int(repro["lr_scheduler"]["warmup_steps"]),
                "total_steps": total_steps,
            }

        return cls(
            datamodule=datamodule,
            student=student,
            activation_snapshot=activation_snapshot,
            layer_pairs=list(layer_pairs),
            optimizer=optimizer,
            alignment_weight=float(alignment_weight),
            alignment_batch_size=int(repro["alignment"]["batch_size"]),
            init_seed=int(repro["seeds"]["global_seed"]),
            lr_scheduler=lr_scheduler,
        )

    @property
    def snapshot(self) -> ActivationSnapshot:
        """The frozen snapshot passed at construction. Reduction and other
        metadata are read through this accessor; tensors live as buffers on
        the module itself (see ``_probe_input_ids`` etc.)."""
        return self._snapshot

    def configure_model(self) -> None:
        """No-op. The student is passed pre-instantiated at ``__init__`` time.

        Override this in a subclass if lazy instantiation is needed (e.g. for
        FSDP). The default path keeps the constructor deterministic and avoids
        the failure modes the previous refactor hit with persistent hooks
        installed at configure_model time.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Seed before optimizer construction. Student weight init is the
        caller's job (before passing the student in)."""
        torch.manual_seed(self.init_seed)

    def forward(self, **inputs: Any) -> Any:
        return self.student(**inputs)

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self(**batch)
        task_loss: Tensor = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        self.log("train_task_loss", task_loss, prog_bar=False, on_step=True, on_epoch=False)

        if self.alignment_weight > 0:
            align_loss = self._alignment_loss()
            self.log(
                "train_align_loss",
                align_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            total = task_loss + self.alignment_weight * align_loss
        else:
            total = task_loss

        self.log(
            "train_total_loss",
            total,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )
        return total

    def _sample_probe_indices(self) -> Tensor:
        """Draw ``alignment_batch_size`` unique indices (or fewer if the probe
        set is smaller) uniformly at random on each call. Uses
        :func:`torch.randperm`, seeded by PyTorch's global RNG - callers who
        need reproducibility should ``torch.manual_seed`` upstream or inside
        a subclass override.
        """
        n_probe = int(self._probe_input_ids.shape[0])
        k = min(self.alignment_batch_size, n_probe)
        # randperm lives on the same device as the probe buffer (CUDA when
        # moved), so we don't induce a host-device sync per step.
        perm = torch.randperm(n_probe, device=self._probe_input_ids.device)
        return perm[:k]

    def _alignment_loss(self) -> Tensor:
        """Compute weighted MSE between student activations and the snapshot's
        pre-pooled targets over a randomly sampled probe mini-batch.

        Uses ``ActivationExtractor.capture`` as a per-call context manager so
        hooks are registered only for the duration of this method and torn
        down on exit. No persistent hooks survive into the next
        ``training_step`` - this was the failure mode of the prior refactor.

        Reduction matches ``self.snapshot.reduction``; the snapshot is the
        single source of truth for how activations were pooled.
        """
        specs = [
            LayerSpec(path=pair["student"], reduce=self.snapshot.reduction)
            for pair in self.layer_pairs
        ]
        extractor = ActivationExtractor(specs, detach=False)

        idx = self._sample_probe_indices()
        probe_ids = self._probe_input_ids[idx]
        probe_mask = self._probe_attention_mask[idx]

        with extractor.capture(self.student):
            # No labels - we only want the forward activations. Some student
            # interfaces require `labels` for loss computation; omitting it
            # means `outputs.loss` is None, which is fine since we discard the
            # return value.
            self.student(input_ids=probe_ids, attention_mask=probe_mask)

        student_acts = extractor.get_activations()

        total = torch.zeros((), device=self._probe_input_ids.device, dtype=torch.float32)
        for pair in self.layer_pairs:
            student_path = pair["student"]
            teacher_path = pair["teacher"]
            weight = float(pair.get("weight", 1.0))

            buffer_name = self._layer_buffer_names[teacher_path]
            target = getattr(self, buffer_name)[idx]
            live = student_acts[student_path]
            total = total + weight * ((live - target.to(live.dtype)) ** 2).mean()

        return total

    def configure_optimizers(self) -> Any:
        lr = float(
            self.optimizer_cfg.get("learning_rate", self.optimizer_cfg.get("lr", 1e-4))
        )
        weight_decay = float(self.optimizer_cfg.get("weight_decay", 0.0))
        betas = tuple(self.optimizer_cfg.get("betas", (0.9, 0.999)))
        eps = float(self.optimizer_cfg.get("eps", 1e-8))

        param_groups = _split_param_groups(
            self.student,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

        if self.lr_scheduler_cfg is None:
            return optimizer

        from transformers import get_linear_schedule_with_warmup

        warmup_steps = int(self.lr_scheduler_cfg.get("warmup_steps", 0))
        total_steps = int(self.lr_scheduler_cfg.get("total_steps", 0))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
