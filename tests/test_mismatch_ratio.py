"""Tests for MismatchRatio metric and _compute_kstar helper."""
import logging

import numpy as np
import pytest

from manylatents.algorithms.latent.multi_dimensional_scaling import MDSModule
from manylatents.algorithms.latent.pca import PCAModule
from manylatents.algorithms.latent.umap import UMAPModule
from manylatents.metrics.mismatch_ratio import (
    MismatchRatio,
    _compute_keff,
    _compute_kstar,
)


class _FakeDataset:
    def __init__(self, data):
        self.data = data


@pytest.fixture
def swissroll_like():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 400)
    X = np.stack([t * np.cos(t), t * np.sin(t), rng.standard_normal(400) * 0.5], axis=1)
    rot = rng.standard_normal((3, 20))
    return (X @ rot).astype(np.float32)


def _embed(module_cls, X, **kwargs):
    m = module_cls(n_components=2, random_state=42, **kwargs)
    emb = m.fit_transform(X)
    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    return m, np.asarray(emb)


def test_native_matches_pre_mode_path(swissroll_like):
    X = swissroll_like
    mod, emb = _embed(UMAPModule, X, neighborhood_size=15)
    r = MismatchRatio(emb, dataset=_FakeDataset(X), module=mod, k=200, mode="native")
    direct = _compute_keff(mod)
    np.testing.assert_allclose(r["k_eff"], direct)


def test_common_kernel_works_on_signed_affinity(swissroll_like):
    """PCA/MDS native k_eff is ~0; common_kernel must track k_kernel."""
    X = swissroll_like
    for cls in (PCAModule, MDSModule):
        _, emb = _embed(cls, X)
        # Create a dummy module for native comparison.
        m = cls(n_components=2, random_state=42)
        m.fit_transform(X)
        r_native = MismatchRatio(emb, dataset=_FakeDataset(X), module=m, k=200, mode="native")
        assert np.median(r_native["k_eff"]) < 1.0, f"{cls.__name__} native k_eff should collapse"

        r_ck = MismatchRatio(
            emb, dataset=_FakeDataset(X), module=m, k=200,
            mode="common_kernel", k_kernel=15,
        )
        assert 12.0 <= np.median(r_ck["k_eff"]) <= 18.0, (
            f"{cls.__name__} common_kernel should track k_kernel=15"
        )


def test_common_kernel_does_not_require_module(swissroll_like):
    X = swissroll_like
    _, emb = _embed(UMAPModule, X, neighborhood_size=15)
    r = MismatchRatio(
        emb, dataset=_FakeDataset(X), module=None, k=200,
        mode="common_kernel", k_kernel=15,
    )
    assert np.isfinite(r["median_v"])
    assert r["k_eff"].shape == (X.shape[0],)


def test_common_kernel_requires_embeddings():
    with pytest.raises(ValueError, match="requires embeddings"):
        MismatchRatio(None, dataset=_FakeDataset(np.zeros((10, 5), dtype=np.float32)),
                      module=None, mode="common_kernel")


def test_unknown_mode_raises(swissroll_like):
    X = swissroll_like
    _, emb = _embed(UMAPModule, X, neighborhood_size=15)
    with pytest.raises(ValueError, match="unknown mode"):
        MismatchRatio(emb, dataset=_FakeDataset(X), module=None, mode="bogus")


def test_saturation_warning_fires(swissroll_like, caplog):
    """When >25% of points saturate at k_max, a warning should be logged."""
    X = swissroll_like
    with caplog.at_level(logging.WARNING, logger="manylatents.metrics.mismatch_ratio"):
        _compute_kstar(X, k_max=50)
    assert any("saturate at k_max" in r.getMessage() for r in caplog.records)


def test_k_max_default_is_500():
    """Regression: default k_max was raised from 200 → 500 to avoid ceiling-pin."""
    import inspect
    sig = inspect.signature(_compute_kstar)
    assert sig.parameters["k_max"].default == 500
    sig = inspect.signature(MismatchRatio)
    assert sig.parameters["k"].default == 500
