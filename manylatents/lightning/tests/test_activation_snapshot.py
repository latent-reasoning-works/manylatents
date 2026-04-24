"""Unit tests for ActivationSnapshot invariants and save/load."""
from __future__ import annotations

import pytest
import torch

from manylatents.lightning.activation_snapshot import (
    ActivationSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
)


def _make_valid_fields(n: int = 4, seq_len: int = 8, hidden: int = 16):
    return {
        "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        "sample_ids": list(range(n)),
        "activations": {"encoder.layer.0": torch.zeros(n, hidden)},
        "reduction": "mean",
    }


def test_construct_happy_path() -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    assert snap.reduction == "mean"
    assert "encoder.layer.0" in snap.activations


def test_post_init_rejects_shape_mismatch_attention_mask() -> None:
    fields = _make_valid_fields(n=4)
    fields["attention_mask"] = torch.ones(3, 8, dtype=torch.long)
    with pytest.raises(ValueError, match=r"attention_mask\.shape\[0\]"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_shape_mismatch_sample_ids() -> None:
    fields = _make_valid_fields(n=4)
    fields["sample_ids"] = [0, 1, 2]
    with pytest.raises(ValueError, match=r"len\(sample_ids\)"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_duplicate_sample_ids() -> None:
    fields = _make_valid_fields(n=4)
    fields["sample_ids"] = [0, 1, 1, 2]
    with pytest.raises(ValueError, match=r"unique"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_wrong_activation_first_dim() -> None:
    fields = _make_valid_fields(n=4, hidden=16)
    fields["activations"] = {"encoder.layer.0": torch.zeros(3, 16)}
    with pytest.raises(ValueError, match=r"activations.*shape\[0\]"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_unknown_reduction() -> None:
    fields = _make_valid_fields()
    fields["reduction"] = "gibberish"
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        ActivationSnapshot(**fields)


def test_post_init_accepts_all_valid_reductions() -> None:
    fields = _make_valid_fields()
    for reduction in ("mean", "last_token", "cls", "first_token", "none"):
        fields_copy = {**fields, "reduction": reduction}
        ActivationSnapshot(**fields_copy)


def test_post_init_rejects_device_mismatch_attention_mask() -> None:
    fields = _make_valid_fields()
    # meta device is always available, avoids CUDA requirement
    fields["attention_mask"] = fields["attention_mask"].to("meta")
    with pytest.raises(ValueError, match=r"attention_mask device"):
        ActivationSnapshot(**fields)


def test_post_init_rejects_device_mismatch_activations() -> None:
    fields = _make_valid_fields()
    fields["activations"] = {
        "encoder.layer.0": fields["activations"]["encoder.layer.0"].to("meta")
    }
    with pytest.raises(ValueError, match=r"activations.*device"):
        ActivationSnapshot(**fields)


def test_len_returns_n_samples() -> None:
    snap = ActivationSnapshot(**_make_valid_fields(n=7))
    assert len(snap) == 7


def test_multiple_layers_all_validated() -> None:
    """Every activation tensor must pass shape+device checks, not just the first."""
    fields = _make_valid_fields(n=4, hidden=16)
    fields["activations"] = {
        "encoder.layer.0": torch.zeros(4, 16),
        "encoder.layer.11": torch.zeros(3, 16),  # wrong
    }
    with pytest.raises(ValueError, match=r"encoder\.layer\.11"):
        ActivationSnapshot(**fields)


def test_frozen_cannot_mutate() -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    with pytest.raises((AttributeError, Exception)):
        snap.reduction = "cls"  # type: ignore[misc]


def test_save_load_roundtrip(tmp_path) -> None:
    fields = _make_valid_fields(n=5, hidden=8)
    fields["input_ids"] = torch.randint(0, 100, (5, 8), dtype=torch.long)
    fields["sample_ids"] = [10, 20, 30, 40, 50]
    fields["activations"] = {
        "encoder.layer.0": torch.randn(5, 8),
        "encoder.layer.11": torch.randn(5, 8),
    }
    snap = ActivationSnapshot(**fields)

    path = tmp_path / "snap.pt"
    snap.save(path)
    loaded = ActivationSnapshot.load(path)

    assert torch.equal(loaded.input_ids, snap.input_ids)
    assert torch.equal(loaded.attention_mask, snap.attention_mask)
    assert loaded.sample_ids == snap.sample_ids
    assert loaded.reduction == snap.reduction
    assert set(loaded.activations.keys()) == set(snap.activations.keys())
    for k in snap.activations:
        assert torch.equal(loaded.activations[k], snap.activations[k])


def test_save_accepts_str_path(tmp_path) -> None:
    snap = ActivationSnapshot(**_make_valid_fields())
    path_str = str(tmp_path / "snap.pt")
    snap.save(path_str)
    loaded = ActivationSnapshot.load(path_str)
    assert len(loaded) == len(snap)


def test_load_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "future.pt"
    snap = ActivationSnapshot(**_make_valid_fields())
    blob = {
        "_version": SNAPSHOT_SCHEMA_VERSION + 99,
        "input_ids": snap.input_ids,
        "attention_mask": snap.attention_mask,
        "sample_ids": snap.sample_ids,
        "activations": snap.activations,
        "reduction": snap.reduction,
    }
    torch.save(blob, str(path))
    with pytest.raises(ValueError, match=r"unknown ActivationSnapshot schema _version"):
        ActivationSnapshot.load(path)


def test_load_rejects_missing_keys(tmp_path) -> None:
    path = tmp_path / "malformed.pt"
    torch.save({"_version": SNAPSHOT_SCHEMA_VERSION, "input_ids": torch.zeros(2, 4)}, str(path))
    with pytest.raises(ValueError, match=r"missing keys"):
        ActivationSnapshot.load(path)


def test_load_rejects_non_dict(tmp_path) -> None:
    path = tmp_path / "weird.pt"
    torch.save(torch.zeros(3), str(path))
    with pytest.raises(ValueError, match=r"expected a dict"):
        ActivationSnapshot.load(path)


def test_load_validates_on_read(tmp_path) -> None:
    """A dict with valid schema but broken invariants should raise __post_init__."""
    path = tmp_path / "broken.pt"
    blob = {
        "_version": SNAPSHOT_SCHEMA_VERSION,
        "input_ids": torch.zeros(4, 8, dtype=torch.long),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
        "sample_ids": [0, 1, 1, 2],  # duplicate — __post_init__ should reject
        "activations": {"encoder.layer.0": torch.zeros(4, 16)},
        "reduction": "mean",
    }
    torch.save(blob, str(path))
    with pytest.raises(ValueError, match=r"unique"):
        ActivationSnapshot.load(path)


# ---- from_model --------------------------------------------------------------

import torch.nn as nn  # noqa: E402


class _TinyBertLike(nn.Module):
    """Minimal transformer-shaped module for from_model tests.

    Forward returns (B, L, H) after a stack of Linear layers on top of an
    embedding. Deterministic under torch.manual_seed.
    """

    def __init__(self, vocab: int = 100, hidden: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


def _make_tiny_inputs(n: int = 4, seq_len: int = 6, vocab: int = 100, seed: int = 0):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab, (n, seq_len), dtype=torch.long)
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    sample_ids = list(range(100, 100 + n))
    return input_ids, attention_mask, sample_ids


def test_from_model_matches_manual_extractor() -> None:
    """from_model must produce identical activations to a direct ActivationExtractor pass."""
    from manylatents.lightning.hooks import ActivationExtractor, LayerSpec

    torch.manual_seed(42)
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()

    layer_paths = ["layers.0", "layers.1"]
    reduction = "mean"

    snap = ActivationSnapshot.from_model(
        model,
        input_ids,
        attention_mask,
        sample_ids,
        layer_paths,
        reduction=reduction,
        batch_size=2,
    )

    model.eval()
    extractor = ActivationExtractor(
        [LayerSpec(path=p, reduce=reduction) for p in layer_paths]
    )
    with torch.no_grad():
        with extractor.capture(model):
            model(input_ids=input_ids, attention_mask=attention_mask)
    manual = extractor.get_activations()

    assert set(snap.activations.keys()) == set(manual.keys())
    for k in manual:
        assert torch.allclose(snap.activations[k], manual[k], atol=1e-6), (
            f"mismatch at layer {k}"
        )


def test_from_model_multiple_layers_captured() -> None:
    model = _TinyBertLike(n_layers=3)
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()
    snap = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.0", "layers.1", "layers.2"],
        reduction="mean",
    )
    assert set(snap.activations.keys()) == {"layers.0", "layers.1", "layers.2"}
    for tensor in snap.activations.values():
        assert tensor.shape == (4, 8)


def test_from_model_batching_equivalent_to_single_pass() -> None:
    """Different batch sizes produce the same result."""
    torch.manual_seed(7)
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs(n=10)

    snap_batched = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids, ["layers.1"],
        reduction="mean", batch_size=3,
    )
    snap_single = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids, ["layers.1"],
        reduction="mean", batch_size=10,
    )
    assert torch.allclose(
        snap_batched.activations["layers.1"],
        snap_single.activations["layers.1"],
        atol=1e-6,
    )


def test_from_model_pool_cls_returns_index_zero() -> None:
    torch.manual_seed(0)
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()

    snap = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.1"], reduction="cls",
    )
    # Manually capture raw layer.1 outputs to compare against [:, 0, :].
    raw_outputs = []
    handle = model.layers[1].register_forward_hook(
        lambda m, i, o: raw_outputs.append(o.detach())
    )
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    raw = torch.cat(raw_outputs, dim=0)
    assert torch.allclose(snap.activations["layers.1"], raw[:, 0, :], atol=1e-6)


def test_from_model_pool_mean_matches_extractor_semantics() -> None:
    """from_model 'mean' matches ActivationExtractor's unmasked tensor.mean(dim=1).

    Note: does NOT exclude padding tokens from the mean — this matches the
    existing collaborator behavior and the old pipeline/ code. Consumers who
    need masked pooling should snapshot with reduction='none' and post-process.
    """
    torch.manual_seed(0)
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()

    snap = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.0"], reduction="mean",
    )
    raw_outputs = []
    handle = model.layers[0].register_forward_hook(
        lambda m, i, o: raw_outputs.append(o.detach())
    )
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    raw = torch.cat(raw_outputs, dim=0)
    assert torch.allclose(snap.activations["layers.0"], raw.mean(dim=1), atol=1e-6)


def test_from_model_rejects_unknown_reduction() -> None:
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()
    with pytest.raises(ValueError, match=r"reduction must be one of"):
        ActivationSnapshot.from_model(
            model, input_ids, attention_mask, sample_ids,
            ["layers.0"], reduction="gibberish",
        )


def test_from_model_restores_training_mode() -> None:
    model = _TinyBertLike()
    model.train()
    assert model.training
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()

    ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.0"], reduction="mean",
    )
    assert model.training, "model.training must be restored after from_model"


def test_from_model_no_hooks_leak_after_return() -> None:
    """ActivationExtractor.capture context manager must tear down all hooks."""
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs()

    ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.0", "layers.1"], reduction="mean",
    )
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0, (
            f"lingering hook on {type(mod).__name__}"
        )


def test_from_model_result_passes_post_init() -> None:
    """Snapshot from from_model must be a fully valid snapshot instance."""
    model = _TinyBertLike()
    input_ids, attention_mask, sample_ids = _make_tiny_inputs(n=5)
    snap = ActivationSnapshot.from_model(
        model, input_ids, attention_mask, sample_ids,
        ["layers.0"], reduction="mean",
    )
    assert len(snap) == 5
    assert snap.reduction == "mean"
    # Reconstruct via save/load to exercise __post_init__ on a roundtrip.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/snap.pt"
        snap.save(path)
        ActivationSnapshot.load(path)  # would raise if invariants broken


def test_from_model_rejects_sample_ids_length_mismatch() -> None:
    model = _TinyBertLike()
    input_ids, attention_mask, _ = _make_tiny_inputs(n=4)
    with pytest.raises(ValueError, match=r"len\(sample_ids\)"):
        ActivationSnapshot.from_model(
            model, input_ids, attention_mask, [0, 1, 2],  # wrong length
            ["layers.0"], reduction="mean",
        )
