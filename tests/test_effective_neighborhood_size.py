"""Tests for EffectiveNeighborhoodSize metric.

Covers two modes:
- ``native``: participation ratio of ``module.affinity()``. Must remain
  bit-identical to the original metric (downstream csvs depend on it).
- ``common_kernel``: participation ratio of a shared kNN Gaussian kernel on
  the embedding — works uniformly across DR families (including PCA, MDS,
  Sammon whose native affinity is a signed Gram matrix).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from manylatents.metrics.effective_neighborhood_size import (
    EffectiveNeighborhoodSize,
    _k_eff_common_kernel,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

N_SMALL = 200


@pytest.fixture(scope="module")
def synthetic_data() -> np.ndarray:
    """Small (N=200, d=8) Gaussian blob mixture — deterministic."""
    rng = np.random.RandomState(0)
    centers = rng.randn(4, 8) * 3.0
    X = np.vstack([
        centers[i] + rng.randn(N_SMALL // 4, 8) * 0.5
        for i in range(4)
    ]).astype(np.float32)
    return X


@pytest.fixture(scope="module")
def synthetic_torch(synthetic_data) -> torch.Tensor:
    return torch.from_numpy(synthetic_data)


def _fit_embed(module_cls, X_torch, **kwargs):
    m = module_cls(n_components=2, random_state=0, **kwargs)
    m.fit(X_torch)
    emb = m.transform(X_torch)
    if torch.is_tensor(emb):
        emb = emb.cpu().numpy()
    return m, np.ascontiguousarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_metric_registered():
    from manylatents.metrics import list_metrics
    names = list_metrics()
    assert (
        "effective_neighborhood_size" in names
        or "k_eff" in names
        or "effective_k" in names
    )


# ---------------------------------------------------------------------------
# native mode: bit-identical parity with original implementation
# ---------------------------------------------------------------------------


def _native_reference(module) -> np.ndarray:
    """Re-implements the original metric code path verbatim.

    If this ever drifts from the implementation the parity test fails — which
    is what we want, because the project guarantees bit-identity for
    ``mode='native'`` across releases.
    """
    W = module.affinity(ignore_diagonal=True, use_symmetric=False)
    row_sum = np.array(W.sum(axis=1)).ravel()
    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = np.asarray(W)
    row_sum_sq = np.array((W_dense ** 2).sum(axis=1)).ravel()
    return np.where(row_sum_sq > 0, (row_sum ** 2) / row_sum_sq, 0.0)


def test_native_default_matches_reference(synthetic_torch, synthetic_data):
    """mode='native' (default) must equal the original formula bit-for-bit."""
    from manylatents.algorithms.latent import TSNEModule
    m, emb = _fit_embed(TSNEModule, synthetic_torch)
    result = EffectiveNeighborhoodSize(embeddings=emb, module=m)
    ref = _native_reference(m)
    np.testing.assert_array_equal(result["k_eff"], ref)
    assert result["mean_k_eff"] == pytest.approx(float(np.mean(ref)))


def test_native_explicit_mode(synthetic_torch):
    """mode='native' passed explicitly must match default."""
    from manylatents.algorithms.latent import TSNEModule
    m, emb = _fit_embed(TSNEModule, synthetic_torch)
    r_default = EffectiveNeighborhoodSize(embeddings=emb, module=m)
    r_native = EffectiveNeighborhoodSize(embeddings=emb, module=m, mode="native")
    np.testing.assert_array_equal(r_default["k_eff"], r_native["k_eff"])


def test_native_requires_module():
    with pytest.raises(ValueError, match="requires a fitted module"):
        EffectiveNeighborhoodSize(embeddings=np.zeros((5, 2)), module=None)


# ---------------------------------------------------------------------------
# common_kernel mode: cross-family parity
# ---------------------------------------------------------------------------


K = 15
TOLERANCE = 0.20  # ±20% of k


@pytest.mark.parametrize("name,module_path,kwargs", [
    ("pca",    "PCAModule",    {}),
    ("mds",    "MDSModule",    {"how": "classic"}),
    ("sammon", "SammonModule", {}),
    ("tsne",   "TSNEModule",   {}),
    ("phate",  "PHATEModule",  {}),
])
def test_common_kernel_cross_family(synthetic_torch, name, module_path, kwargs):
    """common_kernel k_eff should be within ±20% of k on every family."""
    import manylatents.algorithms.latent as _latent
    module_cls = getattr(_latent, module_path)
    try:
        m, emb = _fit_embed(module_cls, synthetic_torch, **kwargs)
    except Exception as e:  # optional deps (phate, etc.) may be missing
        pytest.skip(f"{name}: cannot fit ({type(e).__name__}: {e})")

    result = EffectiveNeighborhoodSize(
        embeddings=emb, module=m, mode="common_kernel", k=K,
    )
    mean_k_eff = result["mean_k_eff"]
    assert abs(mean_k_eff - K) <= TOLERANCE * K, (
        f"{name}: mean_k_eff={mean_k_eff:.2f}, expected ~{K} ± {TOLERANCE*K:.1f}"
    )


def test_common_kernel_ignores_signed_gram(synthetic_torch):
    """common_kernel must NOT call module.affinity() — MDS has a signed Gram
    matrix but common_kernel still has to produce a sane k_eff."""
    from manylatents.algorithms.latent import MDSModule

    m, emb = _fit_embed(MDSModule, synthetic_torch, how="classic")

    # Sanity: the native path collapses on MDS (signed-Gram row-sum cancellation).
    native = EffectiveNeighborhoodSize(embeddings=emb, module=m, mode="native")
    assert native["mean_k_eff"] < 1.0, (
        "expected native mode on MDS to collapse (signed Gram) — if this "
        "no longer holds, the common_kernel justification may need revisiting"
    )

    # Break affinity() loudly: common_kernel must not touch it.
    def _boom(*a, **kw):
        raise AssertionError("common_kernel touched module.affinity()")
    m.affinity = _boom  # type: ignore[assignment]

    result = EffectiveNeighborhoodSize(
        embeddings=emb, module=m, mode="common_kernel", k=K,
    )
    assert abs(result["mean_k_eff"] - K) <= TOLERANCE * K


def test_common_kernel_embedding_directly():
    """common_kernel works on a raw ndarray (no module)."""
    rng = np.random.RandomState(1)
    X = rng.randn(150, 3).astype(np.float32)
    out = EffectiveNeighborhoodSize(
        embeddings=X, module=None, mode="common_kernel", k=K,
    )
    assert abs(out["mean_k_eff"] - K) <= TOLERANCE * K


def test_common_kernel_duplicate_points_safe():
    """Adaptive bandwidth must not divide by zero on duplicate points."""
    X = np.zeros((50, 3), dtype=np.float32)
    X[25:] = 1.0  # two exact-duplicate clusters
    out = EffectiveNeighborhoodSize(
        embeddings=X, module=None, mode="common_kernel", k=10,
    )
    assert not np.any(np.isnan(out["k_eff"]))
    assert not np.any(np.isinf(out["k_eff"]))


def test_common_kernel_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        EffectiveNeighborhoodSize(
            embeddings=np.zeros((5, 2)), module=None, mode="bogus",
        )


def test_common_kernel_shapes_and_keys():
    X = np.random.RandomState(3).randn(80, 4).astype(np.float32)
    out = EffectiveNeighborhoodSize(
        embeddings=X, module=None, mode="common_kernel", k=10,
    )
    assert set(out.keys()) >= {
        "mean_k_eff", "median_k_eff", "std_k_eff",
        "min_k_eff", "max_k_eff", "k_eff",
    }
    assert out["k_eff"].shape == (80,)


def test_common_kernel_helper_matches_metric():
    """Helper and top-level metric must return the same per-row array."""
    X = np.random.RandomState(7).randn(120, 5).astype(np.float32)
    direct = _k_eff_common_kernel(X, k=K)
    via_metric = EffectiveNeighborhoodSize(
        embeddings=X, module=None, mode="common_kernel", k=K,
    )["k_eff"]
    np.testing.assert_allclose(direct, via_metric)
