# tests/metrics/test_subspace_commitment.py
"""Tests for the subspace-commitment order parameter."""
import numpy as np
import pytest


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
    res = SubspaceCommitment(X, dataset=_LabeledDataset(labels))
    assert res["n_groups"] == 4
    assert res["mean"] > 0.4
    assert res["excess"] > 0.2


def test_shuffled_labels_reduce_commitment():
    """Label structure is what's measured: shuffling labels must drop the excess."""
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(1)
    X, labels = _planted(rng)
    true = SubspaceCommitment(X, dataset=_LabeledDataset(labels))
    shuf = SubspaceCommitment(X, dataset=_LabeledDataset(rng.permutation(labels)))
    assert true["excess"] > shuf["excess"] + 0.1


def test_isotropic_data_has_no_excess():
    """Unstructured data: commitment matches the random-bases null (excess ~ 0)."""
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(2)
    X = rng.standard_normal((160, 64))
    labels = np.repeat(np.arange(4), 40)
    res = SubspaceCommitment(X, dataset=_LabeledDataset(labels))
    assert abs(res["excess"]) < 0.05


def test_no_labels_returns_nan():
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(3)
    with pytest.warns(RuntimeWarning):
        res = SubspaceCommitment(rng.standard_normal((20, 8)), dataset=None)
    assert np.isnan(res["mean"]) and res["n_groups"] == 0


def test_single_group_returns_nan():
    from manylatents.metrics import SubspaceCommitment

    rng = np.random.default_rng(4)
    with pytest.warns(RuntimeWarning):
        res = SubspaceCommitment(
            rng.standard_normal((20, 8)), dataset=_LabeledDataset([0] * 20)
        )
    assert np.isnan(res["mean"])


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
    }
