"""Staged training callback for phase2 -> phase3 transitions.

Phase1 (snapshot alignment against a frozen teacher) is expected to have
already happened outside ``trainer.fit`` via ``align_on_snapshot``. This
callback drives the phase2 -> phase3 boundary *inside* ``trainer.fit``:

- ``on_train_start``: freeze parameters whose names match any configured
  prefix (phase2), rebuild the optimizer to drop frozen params.
- ``on_train_batch_end``: once ``global_step == phase3_start_step``, restore
  ``requires_grad`` on previously-frozen params and rebuild the optimizer
  again, preserving Adam state for params that were trainable throughout
  phase2.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
from lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


def _matches(name: str, prefix: str) -> bool:
    """Dot-boundary prefix match.

    Returns True iff ``name`` equals ``prefix`` exactly or starts with
    ``prefix + "."``. This avoids numeric-sibling collisions such as
    ``"layer.1"`` overmatching ``"layer.10"``, ``"layer.11"``, ...
    """
    return name == prefix or name.startswith(prefix + ".")


class StagedTrainingCallback(Callback):
    """Toggle ``requires_grad`` and rebuild the optimizer at phase boundaries.

    Parameters
    ----------
    phase3_start_step:
        Global step at which previously-frozen parameters are unfrozen and
        the optimizer is rebuilt. A value of ``0`` means phase2 is skipped
        entirely and the callback behaves as an identity op.
    frozen_prefixes_phase2:
        List of parameter-name prefixes to freeze during phase2. Matching is
        dot-boundary (see :func:`_matches`). Trailing dots on each prefix are
        stripped on construction so ``"bert.encoder.layer.1"`` and
        ``"bert.encoder.layer.1."`` behave identically.

        **Important**: prefixes match the LightningModule's parameter names,
        which include any nested-module attribute prefix. If your module holds
        the actual trainable model as ``self.student = <hf_model>`` (as
        :class:`~manylatents.algorithms.lightning.distillation.Distillation`
        does), then the callback sees names like ``"student.bert.encoder..."``
        - so your prefix must include the ``"student."`` prefix too. This is
        an honest consequence of PyTorch's ``Module.named_parameters()``
        semantic; the callback has no way to know which attribute holds "the
        real model" and refuses to guess.
    """

    def __init__(
        self,
        phase3_start_step: int,
        frozen_prefixes_phase2: List[str],
    ) -> None:
        super().__init__()
        if phase3_start_step < 0:
            raise ValueError(
                f"phase3_start_step must be >= 0, got {phase3_start_step}"
            )
        self.phase3_start_step = int(phase3_start_step)
        # Strip trailing dots for idempotence. Reject empty prefixes, which
        # would match every parameter and almost certainly be a bug.
        normalized: List[str] = []
        for raw in frozen_prefixes_phase2:
            if not isinstance(raw, str):
                raise TypeError(
                    f"frozen_prefixes_phase2 entries must be str, got {type(raw).__name__}"
                )
            p = raw.rstrip(".")
            if p == "":
                raise ValueError(
                    "Empty prefix in frozen_prefixes_phase2 would freeze every "
                    "parameter; reject as almost certainly a bug."
                )
            normalized.append(p)
        self.frozen_prefixes_phase2: List[str] = normalized
        self._phase3_done: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _names_matching(self, model: torch.nn.Module) -> List[str]:
        return [
            name
            for name, _ in model.named_parameters()
            if any(_matches(name, pref) for pref in self.frozen_prefixes_phase2)
        ]

    def _set_requires_grad(
        self,
        model: torch.nn.Module,
        *,
        freeze: bool,
    ) -> List[str]:
        """Apply ``requires_grad = not freeze`` to matching params.

        Returns the list of parameter names that were toggled.
        """
        touched: List[str] = []
        for name, param in model.named_parameters():
            if any(_matches(name, pref) for pref in self.frozen_prefixes_phase2):
                param.requires_grad = not freeze
                touched.append(name)
        return touched

    @staticmethod
    def _capture_optimizer_state(
        optimizer: torch.optim.Optimizer,
    ) -> Dict[int, Dict[str, Any]]:
        """Snapshot ``optimizer.state`` keyed by ``id(param)``.

        ``copy()`` is shallow — Adam's ``exp_avg``/``exp_avg_sq`` tensors are
        shared rather than duplicated, which is fine because Lightning rebuilds
        the optimizer from scratch and we only need the references to survive
        the interim. The ``step`` counter is a plain int and is captured by
        value via the dict copy.
        """
        return {id(p): dict(state) for p, state in optimizer.state.items()}

    @staticmethod
    def _restore_optimizer_state(
        optimizer: torch.optim.Optimizer,
        state_by_param_id: Dict[int, Dict[str, Any]],
    ) -> None:
        """Restore previously-captured state onto matching params in ``optimizer``.

        Params whose ``id()`` does not appear in ``state_by_param_id`` are left
        with fresh (empty) state — this is the correct behavior for params that
        were frozen during the previous phase and therefore had no Adam state.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                prev = state_by_param_id.get(id(p))
                if prev is not None:
                    optimizer.state[p] = prev

    def _rebuild_optimizer_preserving_state(
        self, trainer: Trainer
    ) -> None:
        """Capture Adam state, ask Lightning to rebuild optimizers, restore state.

        Falls back silently if ``trainer`` has no optimizers yet (e.g. before
        ``fit`` has actually started — defensive; shouldn't happen on the
        documented hooks).
        """
        if not trainer.optimizers:
            return
        old_state = self._capture_optimizer_state(trainer.optimizers[0])
        # Ask Lightning to reconfigure optimizers from the LightningModule's
        # ``configure_optimizers``. This drops all state by default; we restore
        # it below for params that survived.
        trainer.strategy.setup_optimizers(trainer)
        if not trainer.optimizers:
            return
        self._restore_optimizer_state(trainer.optimizers[0], old_state)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.phase3_start_step == 0:
            # Phase2 skipped entirely. No freeze, no optimizer rebuild.
            self._phase3_done = True
            return
        if not self.frozen_prefixes_phase2:
            # Nothing to freeze — treat as an identity op but still mark
            # phase2 as active so the phase3 boundary still fires the rebuild.
            return
        touched = self._set_requires_grad(pl_module, freeze=True)
        logger.info(
            "StagedTrainingCallback: phase2 start, froze %d params matching %r",
            len(touched),
            self.frozen_prefixes_phase2,
        )
        self._rebuild_optimizer_preserving_state(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._phase3_done:
            return
        if trainer.global_step < self.phase3_start_step:
            return
        # Crossed the boundary. Unfreeze and rebuild preserving phase2 state.
        touched = self._set_requires_grad(pl_module, freeze=False)
        logger.info(
            "StagedTrainingCallback: phase3 start at step %d, unfroze %d params",
            int(trainer.global_step),
            len(touched),
        )
        self._rebuild_optimizer_preserving_state(trainer)
        self._phase3_done = True
