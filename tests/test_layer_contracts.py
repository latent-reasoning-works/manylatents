"""Tests for manylatents.testing layer contracts + ActivationExtractor.extract_once."""
import pytest
import torch
import torch.nn as nn

from manylatents.testing import assert_layers_distinct, assert_per_layer_contract


# --- assert_layers_distinct -------------------------------------------------

def test_distinct_layers_pass():
    per_layer = {0: torch.randn(2, 4), 1: torch.randn(2, 4), 2: torch.randn(2, 4)}
    assert_layers_distinct(per_layer)  # should not raise


def test_collapsed_layers_raise():
    shared = torch.randn(2, 4)
    collapsed = {0: shared, 1: shared.clone(), 2: shared.clone()}
    with pytest.raises(AssertionError, match="collapse"):
        assert_layers_distinct(collapsed)


def test_single_layer_is_trivially_ok():
    assert_layers_distinct({0: torch.randn(2, 4)})  # nothing to compare


# --- assert_per_layer_contract ----------------------------------------------

def _good_encoder(x):
    """Distinct per layer AND input-dependent."""
    base = float(sum(map(ord, x)))
    return {l: torch.tensor([[base * (l + 1)]]) for l in (0, 1, 2)}


def _collapsing_encoder(x):
    base = torch.tensor([[float(ord(x[0]))]])
    return {l: base.clone() for l in (0, 1, 2)}


def _input_blind_encoder(x):
    return {l: torch.tensor([[float(l)]]) for l in (0, 1, 2)}


def test_contract_passes_for_good_encoder():
    assert_per_layer_contract(_good_encoder, "MAK", "PAK")


def test_contract_catches_layer_collapse():
    with pytest.raises(AssertionError, match="collapse"):
        assert_per_layer_contract(_collapsing_encoder, "MAK", "PAK")


def test_contract_catches_input_blind_encoder():
    with pytest.raises(AssertionError, match="distinguishing"):
        assert_per_layer_contract(_input_blind_encoder, "MAK", "PAK")


def test_contract_rejects_non_mapping_return():
    with pytest.raises(AssertionError, match="mapping"):
        assert_per_layer_contract(lambda x: torch.zeros(1), "MAK", "PAK")


# --- ActivationExtractor.extract_once ---------------------------------------

def test_extract_once_captures_requested_layers():
    from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    net = Net()
    x = torch.randn(2, 4)
    specs = [LayerSpec("blocks[0]", reduce="none"),
             LayerSpec("blocks[2]", reduce="none")]
    acts = ActivationExtractor.extract_once(net, lambda: net(x), specs)
    assert set(acts) == {"blocks[0]", "blocks[2]"}
    assert acts["blocks[0]"].shape == (2, 4)
    # different blocks → different activations (no accidental aliasing)
    assert not torch.allclose(acts["blocks[0]"], acts["blocks[2]"])


def test_extract_once_removes_hooks():
    from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

    net = nn.Sequential(nn.Linear(4, 4))
    ActivationExtractor.extract_once(
        net, lambda: net(torch.randn(1, 4)), [LayerSpec("0", reduce="none")],
    )
    # No forward hooks should remain registered after the one-shot call.
    assert len(net[0]._forward_hooks) == 0
