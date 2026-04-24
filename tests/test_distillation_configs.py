"""Hydra config smoke tests for distillation/*.yaml.

Verifies that every distillation YAML composes cleanly and carries the
expected ``_target_`` pointers. Does NOT instantiate - instantiation would
require a concrete student and snapshot path at compose time, and the
configs declare those as ``???`` (required overrides).
"""
from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def _compose(name: str):
    # config_module points at manylatents.configs; we use the algorithms/lightning
    # subgroup to pick our distillation variant.
    with initialize_config_module(
        config_module="manylatents.configs", version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                f"algorithms/lightning=distillation/{name}",
                "data=swissroll",
            ],
        )


@pytest.mark.parametrize("name", ["base", "staged", "control_task_only"])
def test_config_composes(name: str) -> None:
    cfg = _compose(name)
    algo = cfg.algorithms.lightning
    assert algo._target_ == (
        "manylatents.algorithms.lightning.distillation.Distillation"
    )
    # activation_snapshot uses the load classmethod
    assert algo.activation_snapshot._target_.endswith("ActivationSnapshot.load")


def test_base_has_zero_alignment_weight() -> None:
    cfg = _compose("base")
    assert cfg.algorithms.lightning.alignment_weight == 0.0


def test_staged_has_unit_alignment_weight() -> None:
    cfg = _compose("staged")
    assert cfg.algorithms.lightning.alignment_weight == 1.0


def test_control_task_only_has_zero_alignment_weight() -> None:
    cfg = _compose("control_task_only")
    assert cfg.algorithms.lightning.alignment_weight == 0.0


def test_staged_inherits_from_base() -> None:
    """Staged config uses `defaults: [base]` — verify inherited fields are present."""
    cfg = _compose("staged")
    algo = cfg.algorithms.lightning
    # Fields present only because they're inherited from base:
    assert "optimizer" in algo
    assert "init_seed" in algo
    assert algo.optimizer.learning_rate == 2e-5
