# manylatents/lightning/tests/test_hf_trainer.py
import pytest
import torch
from manylatents.lightning.hf_trainer import HFTrainerModule, HFTrainerConfig


def test_hf_trainer_module_instantiation():
    """Should instantiate with config."""
    config = HFTrainerConfig(
        model_name_or_path="gpt2",
        learning_rate=2e-5,
    )
    module = HFTrainerModule(config)

    assert module.config == config
    assert module.network is None  # Lazy init


def test_hf_trainer_config_defaults():
    """Config should have sensible defaults."""
    config = HFTrainerConfig(model_name_or_path="gpt2")

    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.0
    assert config.warmup_steps == 0
