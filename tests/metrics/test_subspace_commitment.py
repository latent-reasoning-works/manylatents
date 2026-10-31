# tests/metrics/test_subspace_commitment.py
"""Tests for the subspace-commitment order parameter."""
import numpy as np
import pytest

from manylatents.utils.exceptions import MeasurementUnavailable


class _LabeledDataset:
    def __init__(self, labels):
        self.metadata = np.asarray(labels)


def _planted(rng, n_per=40, K=4, d=64, rank=4, noise=0.05):
    """K groups, each living in its own random rank-`rank` subspace."""
    bases = [np.linalg.qr(rng.standard_normal((d, rank)))[0] for _ in range(K)]
    X, labels = [], []
    for k, B in enumerate(bases):
        X.append(rng.standard_normal((n_per, rank)) @ B.T + noise * rng.standard_normal((n_per, d)))
        labels += [k] * n_per
    return np.concatenate(X), np.asarray(labels)


def test_planted_groups_are_committed():
    """Samples in distinct group subspaces -> high commitment, positive excess."""
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(0)
    X, labels = _planted(rng)
    res = SubspaceCommitment(X, dataset=_LabeledDataset(labels), null_policy="random_bases")
    assert res["n_groups"] == 4
    assert res["mean"] > 0.4
    assert res["excess"] > 0.2


def test_shuffled_labels_reduce_commitment():
    """For planted distinct subspaces, shuffling reduces the random-basis contrast."""
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(1)
    X, labels = _planted(rng)
    true = SubspaceCommitment(X, dataset=_LabeledDataset(labels), null_policy="random_bases")
    shuf = SubspaceCommitment(X, dataset=_LabeledDataset(rng.permutation(labels)), null_policy="random_bases")
    assert true["excess"] > shuf["excess"] + 0.1


def test_isotropic_data_has_no_excess():
    """Unstructured data: commitment matches the random-bases null (excess ~ 0)."""
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(2)
    X = rng.standard_normal((160, 64))
    labels = np.repeat(np.arange(4), 40)
    res = SubspaceCommitment(X, dataset=_LabeledDataset(labels), null_policy="random_bases")
    assert abs(res["excess"]) < 0.05


def test_no_labels_refuses():
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(3)
    with pytest.raises(MeasurementUnavailable, match="no labels"):
        SubspaceCommitment(rng.standard_normal((20, 8)), null_policy="random_bases")


def test_single_group_refuses():
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(4)
    with pytest.raises(MeasurementUnavailable, match="2 usable label groups"):
        SubspaceCommitment(
            rng.standard_normal((20, 8)), dataset=_LabeledDataset([0] * 20),
            null_policy="random_bases",
        )


def test_registry_aliases():
    """Registered under both aliases with the documented defaults."""
    from manylatents.metrics import get_metric_registry

    registry = get_metric_registry()
    assert "subspace_commitment" in registry
    assert "commitment" in registry
    assert registry["subspace_commitment"].func.__name__ == "SubspaceCommitment"
    assert registry["subspace_commitment"].params == {
        "rank": 4,
        "n_null": 3,
        "random_seed": 0,
        "null_policy": "label_permutation",
    }


def _identical_groups(d=2):
    X = np.zeros((8, d))
    X[:, 0] = np.tile([-2., -1., 1., 2.], 2)
    return X, _LabeledDataset(np.repeat([0, 1], 4))


@pytest.mark.parametrize("name", ["SubspaceCommitment", "subspace_commitment", "commitment"])
def test_registry_defaults_to_label_permutation(name):
    from manylatents.metrics import get_metric

    X, ds = _identical_groups()
    result = get_metric(name)(X, dataset=ds, rank=1)
    assert result["excess"] == 0
    assert result["p_value"] == 1
    assert result["null_policy"] == "label_permutation"
    old_contrast = get_metric(name)(X, dataset=ds, rank=1, null_policy="random_bases")
    assert old_contrast["excess"] < 0
    assert old_contrast["null_policy"] == "random_bases"


@pytest.mark.parametrize("kwargs", [{"null_policy": None}, {"null_policy": "automatic"}])
def test_direct_call_refuses_invalid_policy(kwargs):
    from manylatents.metrics import SubspaceCommitment

    X, ds = _identical_groups()
    with pytest.raises(MeasurementUnavailable, match="null_policy"):
        SubspaceCommitment(X, dataset=ds, rank=1, **kwargs)


@pytest.mark.parametrize("d, baseline", [
    (2, 0.10335686140609159),
    (4, 0.4131682499157813),
    (8, 0.6334527190431952),
])
def test_explicit_random_bases_preserves_documented_counterexample(d, baseline):
    """This is an orientation contrast, never a label-independence correction."""
    from manylatents.metrics import SubspaceCommitment

    X, ds = _identical_groups(d)
    result = SubspaceCommitment(X, dataset=ds, rank=1, n_null=3,
                               random_seed=0, null_policy="random_bases")
    assert result["mean"] == 0
    assert result["null_mean"] == pytest.approx(baseline, abs=1e-14)
    assert result["excess"] == pytest.approx(-baseline, abs=1e-14)
    assert result["null_policy"] == "random_bases"
    assert "p_value" not in result


@pytest.mark.parametrize("d", [2, 4, 8])
@pytest.mark.parametrize("kwargs", [{}, {"n_null": 19}, {"null_policy": "label_permutation"}])
def test_default_permutation_identical_groups_is_exactly_zero(d, kwargs):
    from manylatents.metrics import SubspaceCommitment

    X, ds = _identical_groups(d)
    result = SubspaceCommitment(X, dataset=ds, rank=1, **kwargs)
    assert result["mean"] == result["null_mean"] == result["excess"] == 0
    assert result["p_value"] == 1  # Every tied permutation counts in the upper tail.
    assert result["null_policy"] == "label_permutation"


def test_permutation_refits_with_fixed_auxiliary_randomness(monkeypatch):
    import manylatents.metrics.subspace_commitment as metric

    X, labels = _planted(np.random.default_rng(17), n_per=12, K=3, d=8, rank=1)
    # Unequal groups, including a small excluded group: all sizes are preserved.
    X, labels = X[:29], labels[:29]
    labels[-2:] = 3
    original = metric.group_half_bases
    calls = []

    def record(embeddings, perm_labels, rank, rng):
        assert embeddings is X
        calls.append((perm_labels.copy(), rng.bit_generator.state))
        return original(embeddings, perm_labels, rank, rng)

    monkeypatch.setattr(metric, "group_half_bases", record)
    result = metric.SubspaceCommitment(
        X, dataset=_LabeledDataset(labels), rank=1, n_null=9,
        random_seed=8,
    )
    assert len(calls) == 10  # Observed plus B complete split/fit/score evaluations.
    assert result["n_groups"] == 2
    for perm_labels, rng_state in calls:
        np.testing.assert_array_equal(np.sort(perm_labels), np.sort(labels))
        assert rng_state == calls[0][1]
    assert any(not np.array_equal(perm, labels) for perm, _ in calls[1:])

    # Each null value equals a fresh public evaluation of that labeling.
    perm_means = [metric.SubspaceCommitment(
        X, dataset=_LabeledDataset(perm), rank=1, n_null=1, random_seed=8,
        null_policy="random_bases",
    )["mean"] for perm, _ in calls[1:].copy()]
    assert result["null_mean"] == np.mean(perm_means)
    assert result["excess"] == result["mean"] - np.mean(perm_means)
    assert result["p_value"] == (1 + sum(t >= result["mean"] for t in perm_means)) / 10


def test_planted_default_permutation_upper_tail_and_determinism():
    from manylatents.metrics import SubspaceCommitment

    X, labels = _planted(np.random.default_rng(0), n_per=20, K=3, d=16, rank=1)
    kwargs = dict(dataset=_LabeledDataset(labels), rank=1, n_null=19,
                  random_seed=0)
    result = SubspaceCommitment(X, **kwargs)
    assert result == SubspaceCommitment(X, **kwargs)
    assert result["excess"] > 0.2
    assert result["null_policy"] == "label_permutation"
    assert result["p_value"] == 1 / 20  # Plus-one correction even with no exceedances.


@pytest.mark.parametrize("route", ["registry", "hydra"])
@pytest.mark.parametrize("policy_kwargs, policy", [
    ({}, "label_permutation"),
    ({"null_policy": "random_bases"}, "random_bases"),
])
def test_null_provenance_survives_evaluation_saving_and_logging(
    route, policy_kwargs, policy, tmp_path, monkeypatch,
):
    import json
    from unittest.mock import MagicMock

    import pandas as pd
    from omegaconf import OmegaConf

    from manylatents.evaluate import evaluate
    from manylatents.callbacks.embedding.atomic_writer import write_embedding_outputs_atomic
    from manylatents.callbacks.embedding.save_outputs import SaveOutputs
    import manylatents.callbacks.embedding.save_outputs as save_module
    import manylatents.callbacks.embedding.wandb_log_scores as log_module

    X, ds = _identical_groups()
    name = "subspace_commitment"
    if route == "registry":
        scores = evaluate(X, dataset=ds, metrics=[name], rank=1, **policy_kwargs)
    else:
        config = OmegaConf.create({
            "_target_": "manylatents.metrics.subspace_commitment.SubspaceCommitment",
            "_partial_": True, "at": "embedding", "rank": 1, **policy_kwargs,
        })
        scores = evaluate(X, dataset=ds, metrics={name: config})
    assert scores[f"{name}.null_policy"] == policy
    outputs = {"embeddings": X, "scores": scores}

    # Exercise the real CSV and JSON writers, without an external W&B session.
    monkeypatch.setattr(save_module, "wandb", None)
    saver = SaveOutputs(save_dir=str(tmp_path), use_timestamp=False,
                        save_additional_outputs=True, save_metric_tables=True)
    saver.on_latent_end(ds, outputs)
    saved_json = json.loads((tmp_path / "embeddings_experiment_scores.json").read_text())
    assert saved_json == scores
    saved_csv = pd.read_csv(next(tmp_path.glob("metrics_summary_*.csv")))
    assert saved_csv.loc[0, f"{name}.null_policy"] == policy
    assert saved_csv.loc[0, f"{name}.excess"] == pytest.approx(scores[f"{name}.excess"])
    write_embedding_outputs_atomic(outputs, tmp_path / "outputs.json")
    assert json.loads((tmp_path / "outputs.json").read_text())["scores"] == scores

    mock_wandb = MagicMock()
    monkeypatch.setattr(log_module, "wandb", mock_wandb)
    log_module.WandbLogScores().on_latent_end(ds, outputs)
    logged = {key: value for call in mock_wandb.log.call_args_list
              for key, value in call.args[0].items()}
    assert logged == {f"embedding/{key}": value for key, value in scores.items()}

    # The engine also logs scores directly, without WandbLogScores installed.
    from manylatents.data.precomputed_datamodule import PrecomputedDataModule
    import manylatents.experiment as engine

    monkeypatch.setattr(engine, "_load_precomputed_from_datamodule", lambda dm: outputs)
    mock_run = MagicMock()
    result = engine.run_experiment(
        datamodule=PrecomputedDataModule(data=X, batch_size=len(X)),
        algorithm=None, trainer=None, eval_only=True, wandb_run=mock_run,
    )
    assert result["scores"] == scores
    mock_run.log.assert_called_once_with({f"metrics/{key}": value for key, value in scores.items()})
    mock_run.finish.assert_called_once()


@pytest.mark.parametrize("kwargs, reason", [
    ({"rank": 0}, "rank"), ({"rank": -1}, "rank"),
    ({"rank": 1.5}, "rank"), ({"n_null": 0}, "n_null"),
    ({"n_null": True}, "n_null"), ({"random_seed": -1}, "random_seed"),
])
def test_invalid_parameters_refuse(kwargs, reason):
    from manylatents.metrics import SubspaceCommitment

    X, ds = _identical_groups()
    with pytest.raises(MeasurementUnavailable, match=reason):
        SubspaceCommitment(X, dataset=ds, null_policy="label_permutation", **kwargs)


@pytest.mark.parametrize("policy", ["random_bases", "label_permutation"])
@pytest.mark.parametrize("invalid", ["nonfinite", "zero_energy", "misaligned_labels"])
def test_unavailable_evidence_refuses(policy, invalid):
    from manylatents.metrics import SubspaceCommitment

    X, ds = _identical_groups()
    if invalid == "nonfinite":
        X[0, 0] = np.nan
    elif invalid == "zero_energy":
        X[0] = 0  # One undefined sample must not disappear from the average.
    else:
        ds.metadata = ds.metadata[:-1]
    with pytest.raises(MeasurementUnavailable):
        SubspaceCommitment(X, dataset=ds, rank=1, null_policy=policy)


def test_undefined_permutation_is_not_discarded(monkeypatch):
    import manylatents.metrics.subspace_commitment as metric

    X, ds = _identical_groups()
    original = metric.commitment_profile
    count = 0

    def undefined_null(embeddings, bases):
        nonlocal count
        count += 1
        if count > 2:  # Both observed halves succeeded; first permutation fails.
            return np.full(len(embeddings), np.nan)
        return original(embeddings, bases)

    monkeypatch.setattr(metric, "commitment_profile", undefined_null)
    with pytest.raises(MeasurementUnavailable, match="undefined projection energy"):
        metric.SubspaceCommitment(X, dataset=ds, rank=1,
                                  null_policy="label_permutation")
