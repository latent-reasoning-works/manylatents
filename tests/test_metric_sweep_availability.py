"""A parameter sweep preserves each request, including unavailable measurements."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from manylatents.evaluate import evaluate
from manylatents.metrics.trustworthiness import Trustworthiness
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.metrics import flatten_and_unroll_metrics


@pytest.fixture
def evidence():
    data = np.random.default_rng(17).normal(size=(100, 8))
    return data[:, :2], SimpleNamespace(data=data)


@pytest.fixture
def config():
    return OmegaConf.load(
        Path(__file__).parents[1] / "manylatents/configs/metrics/trustworthiness_k.yaml"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_sweep_preserves_every_value_and_reason(config, evidence, reverse):
    if reverse:
        config.trustworthiness.n_neighbors = list(reversed(config.trustworthiness.n_neighbors))
    low, dataset = evidence
    original = OmegaConf.to_container(config)
    scores = evaluate(low, dataset=dataset, metrics=flatten_and_unroll_metrics(config))
    assert OmegaConf.to_container(config) == original
    assert set(scores) == {f"trustworthiness__n_neighbors_{k}" for k in (15, 25, 50, 100, 250)}
    for k in (15, 25, 50):
        assert scores[f"trustworthiness__n_neighbors_{k}"] == pytest.approx(
            Trustworthiness(low, dataset=dataset, n_neighbors=k)
        )
    for k in (100, 250):
        unavailable = scores[f"trustworthiness__n_neighbors_{k}"]
        assert isinstance(unavailable, MeasurementUnavailable)
        assert f"k={k}" in str(unavailable)
        assert "n=100" in str(unavailable)
        with pytest.raises(TypeError):
            float(unavailable)


@pytest.mark.parametrize("ks", [[100, 250], [100], []])
@pytest.mark.parametrize("other_metric", [False, True])
def test_wholly_invalid_sweep_refuses_even_if_another_metric_succeeds(config, evidence, ks, other_metric):
    config.trustworthiness.n_neighbors = ks
    if other_metric:
        # Do not infer group membership from the user-defined metric name.
        config["trustworthiness__unrelated"] = {
            "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
            "_partial_": True, "n_neighbors": 15,
        }
    low, dataset = evidence
    with pytest.raises(MeasurementUnavailable, match="sweep 'trustworthiness'") as failure:
        evaluate(low, dataset=dataset, metrics=flatten_and_unroll_metrics(config))
    for k in ks:
        assert f"k={k}" in str(failure.value)


def test_sweep_does_not_swallow_programming_errors(config, evidence, monkeypatch):
    def broken_metric(*args, **kwargs):
        raise ValueError("implementation bug")

    monkeypatch.setattr("manylatents.metrics.trustworthiness.Trustworthiness", broken_metric)
    low, dataset = evidence
    with pytest.raises(ValueError, match="implementation bug") as failure:
        evaluate(low, dataset=dataset, metrics=flatten_and_unroll_metrics(config))
    assert type(failure.value) is ValueError


def test_cartesian_sweep_preserves_missing_output_entries(config, evidence):
    config.trustworthiness.n_neighbors = [15, 100]
    config.trustworthiness.at = ["embedding", "absent_output"]
    low, dataset = evidence
    scores = evaluate(low, dataset=dataset, metrics=flatten_and_unroll_metrics(config))
    assert len(scores) == 4
    assert sum(isinstance(value, float) for value in scores.values()) == 1
    for name, value in scores.items():
        if "absent_output" in name:
            assert isinstance(value, MeasurementUnavailable)
            assert name in str(value) and "absent_output" in str(value)


def test_partial_sweep_survives_json_csv_and_logging(config, evidence, tmp_path, monkeypatch):
    from manylatents.callbacks.embedding.atomic_writer import write_embedding_outputs_atomic
    from manylatents.callbacks.embedding.save_outputs import SaveOutputs
    from manylatents.callbacks.embedding.wandb_log_scores import WandbLogScores
    import pandas as pd

    low, dataset = evidence
    scores = evaluate(low, dataset=dataset, metrics=flatten_and_unroll_metrics(config))
    outputs = {"embeddings": low, "scores": scores}
    write_embedding_outputs_atomic(outputs, tmp_path / "outputs.json")
    saved = json.loads((tmp_path / "outputs.json").read_text())["scores"]
    callback = SaveOutputs(save_dir=str(tmp_path), save_additional_outputs=True,
                           save_metric_tables=True, use_timestamp=False)
    callback.on_latent_end(dataset, outputs)
    csv_scores = pd.read_csv(next(tmp_path.glob("metrics_summary_*.csv"))).iloc[0]
    additional = json.loads(next(tmp_path.glob("embeddings_*_scores.json")).read_text())
    mock_wandb = MagicMock()
    monkeypatch.setattr("manylatents.callbacks.embedding.wandb_log_scores.wandb", mock_wandb)
    WandbLogScores().on_latent_end(dataset, outputs)
    logged = {key: value for call in mock_wandb.log.call_args_list
              for key, value in call.args[0].items()}
    assert set(saved) == set(additional) == set(csv_scores.index) == set(scores)
    for k in (100, 250):
        name = f"trustworthiness__n_neighbors_{k}"
        expected = {"status": "unavailable", "reason": str(scores[name])}
        assert saved[name] == additional[name] == json.loads(csv_scores[name]) == expected
        assert logged[f"embedding/{name}"] == expected
    for k in (15, 25, 50):
        name = f"trustworthiness__n_neighbors_{k}"
        assert saved[name] == additional[name] == scores[name]
        assert float(csv_scores[name]) == pytest.approx(scores[name])
        assert logged[f"embedding/{name}"] == scores[name]
