"""Tests for the Fold + PreservationScale diagnostic metrics.
Copy to manylatents `tests/` when applying to a clean clone (see APPLY.md)."""
import numpy as np
from types import SimpleNamespace
from manylatents.metrics import list_metrics
from manylatents.metrics.registry import get_metric
from manylatents.metrics.preservation_diagnostics import Fold, PreservationScale


def test_fold_registered():
    assert "fold" in list_metrics()


def test_preservation_scale_registered():
    assert "preservation_scale" in list_metrics()


def test_fold_higher_on_folded_than_flat():
    rng = np.random.default_rng(0)
    flat2 = rng.uniform(0, 1, (300, 2))
    flat = np.column_stack([flat2, np.zeros((300, 8))])
    out_flat = Fold(embeddings=np.zeros((300, 2)),
                    dataset=SimpleNamespace(data=flat), k_ref=20, k_graph=8)
    assert out_flat["fold"].shape == (300,)
    assert np.nanmean(out_flat["fold"]) < 0.3
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, 300)
    h = rng.uniform(0, 10, 300)
    roll = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    roll = np.column_stack([roll, np.zeros((300, 7))])
    out_roll = Fold(embeddings=np.zeros((300, 2)),
                    dataset=SimpleNamespace(data=roll), k_ref=20, k_graph=8)
    assert np.nanmean(out_roll["fold"]) > np.nanmean(out_flat["fold"])


def test_preservation_scale_smaller_for_faithful_embedding():
    rng = np.random.default_rng(1)
    X2 = rng.uniform(0, 1, (300, 2))
    X = np.column_stack([X2, np.zeros((300, 8))])
    ds = SimpleNamespace(data=X)
    faithful = PreservationScale(embeddings=X2, dataset=ds, k_graph=8, k_max=100)
    random = PreservationScale(embeddings=rng.standard_normal((300, 2)), dataset=ds,
                               k_graph=8, k_max=100)
    assert faithful["s_star"].shape == (300,)
    assert np.nanmedian(faithful["s_star"]) < np.nanmedian(random["s_star"])


def test_metrics_resolve_through_registry():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.uniform(0, 1, (200, 2)), np.zeros((200, 8))])
    ds = SimpleNamespace(data=X)
    rf = get_metric("fold")(embeddings=np.zeros((200, 2)), dataset=ds)
    rs = get_metric("preservation_scale")(embeddings=rng.standard_normal((200, 2)), dataset=ds)
    assert "fold" in rf and "s_star" in rs
