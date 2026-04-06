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


def test_hf_trainer_config_output_hidden_states_default():
    """output_hidden_states should default to False."""
    config = HFTrainerConfig(model_name_or_path="gpt2")
    assert config.output_hidden_states is False


@pytest.mark.slow
def test_hf_trainer_output_hidden_states():
    """forward() with output_hidden_states=True returns hidden state tuple."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        output_hidden_states=True,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world", "Test input"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )

    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.hidden_states is not None
    # hidden_states is a tuple of (n_layers + 1) tensors (embedding + each layer)
    n_layers = module.network.config.num_hidden_layers
    assert len(outputs.hidden_states) == n_layers + 1
    # Each tensor: (batch, seq_len, hidden_dim)
    assert outputs.hidden_states[0].shape[0] == 2  # batch_size


@pytest.mark.slow
def test_hf_trainer_output_hidden_states_disabled():
    """forward() with output_hidden_states=False returns None for hidden_states."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        output_hidden_states=False,
    )
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

    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.hidden_states is None
