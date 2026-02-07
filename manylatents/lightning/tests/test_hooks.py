# manylatents/lightning/tests/test_hooks.py
import pytest
import torch
import torch.nn as nn
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec, resolve_layer


class MockTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(64, 64)
        self.mlp = nn.Linear(64, 64)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.layers = nn.ModuleList([MockTransformerBlock() for _ in range(4)])
        self.lm_head = nn.Linear(64, 100)


def test_layer_spec_defaults():
    spec = LayerSpec(path="model.layers[-1]")
    assert spec.path == "model.layers[-1]"
    assert spec.extraction_point == "output"
    assert spec.reduce == "mean"


def test_layer_spec_custom():
    spec = LayerSpec(
        path="model.layers[12].self_attn",
        extraction_point="hidden_states",
        reduce="last_token",
    )
    assert spec.extraction_point == "hidden_states"
    assert spec.reduce == "last_token"


def test_layer_spec_invalid_reduce():
    with pytest.raises(ValueError, match="reduce must be one of"):
        LayerSpec(path="model.layers[-1]", reduce="invalid")


# resolve_layer tests


def test_resolve_layer_simple():
    model = MockModel()
    layer = resolve_layer(model, "lm_head")
    assert layer is model.lm_head


def test_resolve_layer_nested():
    model = MockModel()
    layer = resolve_layer(model, "layers[2].self_attn")
    assert layer is model.layers[2].self_attn


def test_resolve_layer_negative_index():
    model = MockModel()
    layer = resolve_layer(model, "layers[-1]")
    assert layer is model.layers[-1]


def test_resolve_layer_invalid():
    model = MockModel()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        resolve_layer(model, "nonexistent")


# ActivationExtractor tests


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


def test_activation_extractor_single_layer():
    model = SimpleModel()
    spec = LayerSpec(path="layer1", reduce="none")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert "layer1" in activations
    assert activations["layer1"].shape == (4, 20)


def test_activation_extractor_multiple_layers():
    model = SimpleModel()
    specs = [
        LayerSpec(path="layer1", reduce="none"),
        LayerSpec(path="layer2", reduce="none"),
    ]
    extractor = ActivationExtractor(specs)

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert len(activations) == 2
    assert activations["layer1"].shape == (4, 20)
    assert activations["layer2"].shape == (4, 5)


def test_activation_extractor_clears_after_get():
    model = SimpleModel()
    spec = LayerSpec(path="layer1", reduce="none")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    _ = extractor.get_activations()

    # Second call should return empty (already cleared)
    activations2 = extractor.get_activations()
    assert len(activations2) == 0


# Sequence reduction tests


class SequenceModel(nn.Module):
    """Model that outputs (batch, seq_len, dim)."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        # x: (batch, seq_len, 10) -> (batch, seq_len, 20)
        return self.layer(x)


def test_activation_extractor_reduce_mean():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="mean")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)  # batch=4, seq_len=8, dim=10

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 20)  # Reduced over seq_len


def test_activation_extractor_reduce_last_token():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="last_token")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 20)


def test_activation_extractor_reduce_all():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="all")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 8, 20)  # Kept full sequence
