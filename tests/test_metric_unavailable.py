"""An unavailable metric preserves other scores and algorithm outputs."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.api import run
from manylatents.evaluate import evaluate
from manylatents.utils.exceptions import MeasurementUnavailable


@pytest.fixture
def collapsed():
    return np.tile(np.arange(10, dtype=np.float32).reshape(5, 2), (20, 1))


def assert_scores(scores):
    assert set(scores) == {"trustworthiness", "lid"}
    assert isinstance(scores["trustworthiness"], float)
    assert np.isfinite(scores["trustworthiness"])
    unavailable = scores["lid"]
    assert isinstance(unavailable, MeasurementUnavailable)
    assert "only 5 distinct points for k=20" in str(unavailable)
    assert unavailable.__traceback__ is None


@pytest.mark.parametrize("reverse", [False, True])
def test_registry_preserves_other_metrics(collapsed, reverse):
    metrics = ["trustworthiness", "lid"]
    if reverse:
        metrics.reverse()
    scores = evaluate(collapsed, dataset=SimpleNamespace(data=collapsed), metrics=metrics)
    assert_scores(scores)


def test_hydra_preserves_other_metrics(collapsed):
    metrics = {
        name: OmegaConf.create({"_target_": target, "_partial_": True})
        for name, target in {
            "lid": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "trustworthiness": "manylatents.metrics.trustworthiness.Trustworthiness",
        }.items()
    }
    assert_scores(evaluate(collapsed, dataset=SimpleNamespace(data=collapsed), metrics=metrics))


def test_registry_does_not_swallow_programming_errors(collapsed, monkeypatch):
    from manylatents.metrics.registry import get_metric

    def broken_metric(**kwargs):
        raise ValueError("implementation bug")

    monkeypatch.setattr(get_metric("lid"), "func", broken_metric)
    with pytest.raises(ValueError, match="implementation bug") as failure:
        evaluate(collapsed, metrics=["lid"])
    assert type(failure.value) is ValueError


class CollapsingAlgorithm(LatentModule):
    def fit(self, x, y=None):
        self._is_fitted = True

    def transform(self, x):
        return torch.arange(10, dtype=torch.float32).reshape(5, 2).repeat(len(x) // 5, 1)


def test_run_preserves_embedding_and_scores(collapsed):
    result = run(input_data=collapsed, algorithm=CollapsingAlgorithm(),
                 metrics=["lid", "trustworthiness"])
    np.testing.assert_array_equal(result["embeddings"], collapsed)
    assert_scores(result["scores"])


def test_unavailable_survives_save_and_logging(collapsed, tmp_path, monkeypatch):
    from manylatents.api import _resolve_datamodule
    from manylatents.callbacks.embedding.atomic_writer import (
        write_embedding_outputs_atomic,
    )
    from manylatents.callbacks.embedding.save_outputs import SaveOutputs
    from manylatents.callbacks.embedding.wandb_log_scores import WandbLogScores
    from manylatents.experiment import run_experiment

    mock_wandb = MagicMock()
    monkeypatch.setattr("manylatents.callbacks.embedding.wandb_log_scores.wandb", mock_wandb)
    wandb_run = MagicMock()
    result = run_experiment(
        datamodule=_resolve_datamodule(input_data=collapsed),
        algorithm=CollapsingAlgorithm(), trainer=None, metrics=["lid", "trustworthiness"],
        wandb_run=wandb_run,
        embedding_callbacks=[
            SaveOutputs(save_dir=str(tmp_path), save_additional_outputs=True,
                        save_metric_tables=True, use_timestamp=False),
            WandbLogScores(),
        ],
    )
    assert_scores(result["scores"])
    write_embedding_outputs_atomic(result, tmp_path / "outputs.json")
    saved = json.loads((tmp_path / "outputs.json").read_text())["scores"]
    assert tmp_path.is_dir()
    csv_scores = pd.read_csv(next(tmp_path.glob("metrics_summary_*.csv"))).iloc[0]
    additional = json.loads(next(tmp_path.glob("embeddings_*_scores.json")).read_text())
    expected = result["scores"]["lid"].to_dict()
    assert saved["lid"] == additional["lid"] == json.loads(csv_scores["lid"]) == expected
    for logger, prefix in [(mock_wandb, "single_algorithm"), (wandb_run, "metrics")]:
        logged = {key: value for call in logger.log.call_args_list
                  for key, value in call.args[0].items()}
        assert logged[f"{prefix}/lid"] == expected
        assert logged[f"{prefix}/trustworthiness"] == result["scores"]["trustworthiness"]


def test_plot_unavailable_metric_uses_labels(collapsed, tmp_path):
    from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings

    labels = np.arange(len(collapsed)) % 5
    callback = PlotEmbeddings(save_dir=str(tmp_path), color_by="lid")
    colors = callback._get_color_array(None, {
        "scores": {"lid": MeasurementUnavailable("too few distinct points")},
        "label": labels,
    })
    np.testing.assert_array_equal(colors, labels)
