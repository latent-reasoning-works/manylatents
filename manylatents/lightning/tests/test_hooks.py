# manylatents/lightning/tests/test_hooks.py
import pytest
from manylatents.lightning.hooks import LayerSpec


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
