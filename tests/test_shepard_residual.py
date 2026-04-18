"""Tests for ShepardResidual metric."""
import numpy as np
import pytest

from manylatents.metrics.shepard_residual import ShepardResidual


class _StubDataset:
    def __init__(self, data):
        self.data = data


def _make_swiss_roll(n=400, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(0, 10, n)
    x = t * np.cos(t)
    z = t * np.sin(t)
    data = np.column_stack([x, h, z]).astype(np.float32)
    if noise > 0:
        data = data + rng.normal(0, noise, size=data.shape).astype(np.float32)
    # Embedding: unrolled (t, h)
    embedding = np.column_stack([t, h]).astype(np.float32)
    return data, embedding


class TestShepardResidual:
    def test_returns_required_keys(self):
        data, embedding = _make_swiss_roll(n=200)
        out = ShepardResidual(embedding, dataset=_StubDataset(data), k=10)
        for key in (
            "residual",
            "alpha",
            "mean_residual",
            "median_residual",
            "std_residual",
            "mean_residual_normalized",
        ):
            assert key in out

    def test_per_point_shape(self):
        data, embedding = _make_swiss_roll(n=200)
        out = ShepardResidual(embedding, dataset=_StubDataset(data), k=10)
        assert out["residual"].shape == (200,)

    def test_alpha_positive(self):
        data, embedding = _make_swiss_roll(n=200)
        out = ShepardResidual(embedding, dataset=_StubDataset(data), k=10)
        assert out["alpha"] > 0

    def test_residuals_nonnegative(self):
        data, embedding = _make_swiss_roll(n=200)
        out = ShepardResidual(embedding, dataset=_StubDataset(data), k=10)
        assert (out["residual"] >= 0).all()

    def test_isometric_low_residual(self):
        # Identity embedding (perfect ambient = embedding) → near-zero residual.
        rng = np.random.default_rng(0)
        data = rng.standard_normal((300, 5)).astype(np.float32)
        embedding = data.copy()
        out = ShepardResidual(embedding, dataset=_StubDataset(data), k=10)
        # alpha ~ 1, residuals ~ 0 (machine precision).
        assert abs(out["alpha"] - 1.0) < 1e-3
        assert out["mean_residual"] < 1e-3

    def test_distorted_higher_residual(self):
        # Distorted embedding (random scrambling) should yield larger residuals
        # than the perfect identity embedding on the same data.
        rng = np.random.default_rng(0)
        data = rng.standard_normal((300, 5)).astype(np.float32)
        emb_good = data.copy()
        emb_bad = rng.standard_normal((300, 2)).astype(np.float32)

        out_good = ShepardResidual(emb_good, dataset=_StubDataset(data), k=10)
        out_bad = ShepardResidual(emb_bad, dataset=_StubDataset(data), k=10)

        assert out_bad["mean_residual"] > out_good["mean_residual"]

    def test_requires_dataset(self):
        rng = np.random.default_rng(0)
        embedding = rng.standard_normal((200, 2)).astype(np.float32)
        with pytest.raises(ValueError):
            ShepardResidual(embedding, dataset=None, k=10)

    def test_dataset_without_data_attr_rejected(self):
        rng = np.random.default_rng(0)
        embedding = rng.standard_normal((200, 2)).astype(np.float32)

        class _NoData:
            pass

        with pytest.raises(ValueError):
            ShepardResidual(embedding, dataset=_NoData(), k=10)

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 5)).astype(np.float32)
        embedding = rng.standard_normal((150, 2)).astype(np.float32)
        with pytest.raises(ValueError):
            ShepardResidual(embedding, dataset=_StubDataset(data), k=10)

    def test_registered_under_aliases(self):
        from manylatents.metrics import get_metric

        for alias in ("shepard_residual", "shepard"):
            spec = get_metric(alias)
            assert spec is not None
