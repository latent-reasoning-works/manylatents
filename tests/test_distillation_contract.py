"""Regression coverage for the retained distillation/debug boundary."""
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from manylatents.algorithms.lightning.distillation import Distillation
from manylatents.callbacks.debug import FirstBatchLoggerCallback, attach_nan_detector
from manylatents.lightning.activation_snapshot import ActivationSnapshot


def make_distillation(n=4, **kwargs):
    snapshot = ActivationSnapshot(
        input_ids=torch.zeros(n, 2, dtype=torch.long),
        attention_mask=torch.ones(n, 2, dtype=torch.long),
        sample_ids=list(range(n)),
        activations={"layer": torch.zeros(n, 2)},
        reduction="mean",
    )
    return Distillation(
        datamodule=None, student=nn.Sequential(nn.Linear(2, 2)),
        activation_snapshot=snapshot,
        layer_pairs=[{"student": "0", "teacher": "layer"}],
        optimizer={}, alignment_weight=1, **kwargs,
    )


@pytest.mark.parametrize("requested", [0, -1, 3.9, True, "3", 5])
def test_alignment_batch_request_is_never_coerced_or_clamped(requested):
    with pytest.raises(ValueError, match="alignment_batch_size"):
        make_distillation(alignment_batch_size=requested)._sample_probe_indices()


@pytest.mark.parametrize("n", [1, 4, 20])
def test_absent_alignment_batch_uses_available_default(n):
    indices = make_distillation(n=n)._sample_probe_indices()
    assert len(indices) == min(16, n)
    assert len(indices.unique()) == len(indices)


def test_empty_probe_set_has_no_alignment_measurement():
    with pytest.raises(ValueError, match="probe"):
        make_distillation(n=0)._sample_probe_indices()


@pytest.mark.parametrize("factory", [
    lambda value: FirstBatchLoggerCallback(n_steps=value),
    lambda value: FirstBatchLoggerCallback(vocab_size=value),
    lambda value: attach_nan_detector(nn.Identity(), max_reports=value),
])
@pytest.mark.parametrize("value", [-1, 1.9, True, "2"])
def test_debug_parameters_are_validated(factory, value):
    with pytest.raises(ValueError):
        factory(value)


def test_vocab_size_must_be_positive():
    with pytest.raises(ValueError, match="vocab_size"):
        FirstBatchLoggerCallback(vocab_size=0)


def test_negative_token_id_is_out_of_bounds(capsys):
    callback = FirstBatchLoggerCallback(n_steps=0, vocab_size=10)
    callback.on_train_batch_end(None, None, None, {"input_ids": torch.tensor([-1, 2])}, 0)
    assert "oob=True" in capsys.readouterr().out


@pytest.mark.parametrize("ids", [torch.tensor([]), torch.tensor([1.9, 2.0]), torch.tensor([float("nan")])])
def test_unmeasurable_token_bounds_have_reason(ids, capsys):
    callback = FirstBatchLoggerCallback(n_steps=0, vocab_size=10)
    callback.on_train_batch_end(None, None, None, {"input_ids": ids}, 0)
    output = capsys.readouterr().out
    assert "'status': 'unavailable'" in output
    assert "'reason':" in output
    assert "oob=False" not in output


@pytest.mark.parametrize("loss", [None, "bad", torch.ones(2), float("nan"), float("inf")])
def test_unmeasurable_loss_has_explicit_status_and_reason(loss, capsys):
    callback = FirstBatchLoggerCallback(n_steps=1)
    callback.on_train_batch_end(SimpleNamespace(global_step=0), None, {"loss": loss}, {}, 0)
    output = capsys.readouterr().out
    assert "loss={'status': 'unavailable', 'reason':" in output
    assert "loss=nan" not in output
    assert "loss=inf" not in output


def test_unexpected_loss_conversion_failure_propagates():
    class BrokenLoss:
        def __float__(self):
            raise RuntimeError("unexpected failure")

    with pytest.raises(RuntimeError, match="unexpected failure"):
        FirstBatchLoggerCallback().on_train_batch_end(
            SimpleNamespace(global_step=0), None, BrokenLoss(), {}, 0,
        )


def test_all_nonfinite_activations_have_no_finite_maximum(capsys):
    model = nn.Identity()
    handles = attach_nan_detector(model)
    try:
        model(torch.tensor([float("nan"), float("inf")]))
    finally:
        for handle in handles:
            handle.remove()
    output = capsys.readouterr().out
    assert "finite_abs_max={'status': 'unavailable', 'reason':" in output
    assert "finite_abs_max=nan" not in output


def test_finite_activation_subset_is_explicitly_labelled(capsys):
    model = nn.Identity()
    handles = attach_nan_detector(model)
    try:
        model(torch.tensor([float("nan"), -3.0, float("inf")]))
    finally:
        for handle in handles:
            handle.remove()
    output = capsys.readouterr().out
    assert "finite_abs_max=3.0000e+00" in output
    assert "finite_count=1 total_count=3" in output


@pytest.mark.parametrize("seed", [1.9, True, "2"])
def test_seed_is_not_silently_coerced(seed):
    with pytest.raises(ValueError, match="init_seed"):
        make_distillation(init_seed=seed)


@pytest.mark.parametrize("scheduler", [
    {"warmup_steps": 2.9, "total_steps": 10},
    {"warmup_steps": 2, "total_steps": 10.9},
    {"warmup_steps": True, "total_steps": 10},
    {"warmup_steps": -1, "total_steps": 10},
    {"total_steps": 0},
])
def test_schedule_counts_are_not_silently_coerced(scheduler, monkeypatch):
    # Test our argument boundary, without requiring the optional HF package.
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        get_linear_schedule_with_warmup=lambda *args, **kwargs: object(),
    ))
    with pytest.raises(ValueError, match="steps"):
        make_distillation(lr_scheduler=scheduler).configure_optimizers()


def test_scheduler_absent_warmup_defaults_to_zero(monkeypatch):
    received = {}

    def capture_schedule(optimizer, **kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        get_linear_schedule_with_warmup=capture_schedule,
    ))
    make_distillation(lr_scheduler={"total_steps": 10}).configure_optimizers()
    assert received == {"num_warmup_steps": 0, "num_training_steps": 10}


def test_alignment_unavailable_uses_shared_serialization():
    from manylatents.utils.exceptions import MeasurementUnavailable

    with pytest.raises(MeasurementUnavailable) as caught:
        make_distillation(alignment_batch_size=5)._sample_probe_indices()
    assert caught.value.to_dict() == {
        "status": "unavailable", "reason": str(caught.value),
    }
