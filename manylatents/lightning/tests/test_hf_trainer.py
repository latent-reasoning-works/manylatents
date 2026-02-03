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


@pytest.mark.slow
def test_hf_trainer_module_forward_pass():
    """Integration test with actual tiny model."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",  # ~2MB model
        trust_remote_code=True,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    # Create dummy batch
    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world", "Test input"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    # Forward pass
    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.loss is not None
    assert outputs.logits is not None


@pytest.mark.slow
def test_hf_trainer_module_training_step():
    """Test training step computes loss."""
    config = HFTrainerConfig(model_name_or_path="sshleifer/tiny-gpt2")
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    loss = module.training_step(batch, 0)

    assert loss is not None
    assert loss.requires_grad


@pytest.mark.slow
def test_hf_trainer_with_activation_extractor():
    """Verify ActivationExtractor works with HF models - critical integration test."""
    from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

    config = HFTrainerConfig(model_name_or_path="sshleifer/tiny-gpt2")
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create probe batch
    batch = tokenizer(
        ["Hello world", "Test input", "Another sample"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )

    # Extract from last transformer block - use the actual HF path
    # For GPT2: model.transformer.h[-1] is the last block
    spec = LayerSpec(path="transformer.h[-1]", reduce="mean")
    extractor = ActivationExtractor([spec])

    module.eval()
    with torch.no_grad():
        with extractor.capture(module.network):
            _ = module.network(**batch)

    activations = extractor.get_activations()

    assert "transformer.h[-1]" in activations
    # Should have (batch_size, hidden_dim) after mean reduction
    assert activations["transformer.h[-1]"].shape[0] == 3  # 3 samples
    assert len(activations["transformer.h[-1]"].shape) == 2  # (batch, hidden)
