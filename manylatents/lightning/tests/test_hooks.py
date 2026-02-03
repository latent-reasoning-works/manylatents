# manylatents/lightning/tests/test_hooks.py
import pytest
import torch.nn as nn
from manylatents.lightning.hooks import LayerSpec, resolve_layer


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
