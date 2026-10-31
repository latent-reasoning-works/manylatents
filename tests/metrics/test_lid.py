"""Pin the numerical contract of LocalIntrinsicDimensionality (issue #304).

Exercise scale invariance and the declared distinct-neighbor policy without
copying the estimator into the tests.

7.6e-16 is a real activation RMS (Evo2-7B blocks.31), used throughout as the
rescale factor so the invariance tests are anchored to something observed.
"""

import numpy as np
import pytest

from manylatents.metrics.lid import LocalIntrinsicDimensionality as lid
from manylatents.metrics.registry import get_metric

TINY = 7.6e-16


def _subspace_cloud(n=600, intrinsic=5, ambient=64, seed=0):
    """A cloud with a known intrinsic dimension well below its ambient one."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, intrinsic)) @ rng.normal(size=(intrinsic, ambient))).astype(np.float32)


# ---- the estimator still estimates ----

def test_recovers_the_dimension_of_a_linear_subspace():
    assert 3.0 < lid(_subspace_cloud(), k=20) < 8.0


@pytest.mark.parametrize("intrinsic, ambient", [(2, 8), (3, 12)])
def test_uniform_subspace_recovers_known_dimension(intrinsic, ambient):
    from manylatents.api import run

    rng = np.random.default_rng(42)
    basis, _ = np.linalg.qr(rng.normal(size=(ambient, intrinsic)))
    # Uniform on a unit cube in the intrinsic coordinates; an orthonormal
    # embedding preserves its Euclidean geometry in the higher-dimensional space.
    x = rng.uniform(size=(5000, intrinsic)) @ basis.T
    result = run(
        input_data=x,
        algorithm="pca",
        n_components=intrinsic,
        metrics={"lid": {
            "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
            "_partial_": True,
            "at": "dataset",  # measure the original ambient-dimensional cloud
            "k": 40,
        }},
    )
    estimate, = result["scores"].values()
    # Finite neighborhoods have sampling variation and cube boundary bias.
    # A 5% allowance tests dimension recovery without assuming exact unbiasedness.
    assert estimate == pytest.approx(intrinsic, rel=0.05)


def test_per_sample_returns_one_value_per_input_row():
    x = np.repeat(_subspace_cloud(n=200), 3, axis=0)
    out = lid(x, k=20, return_per_sample=True)
    assert out.shape == (600,)
    assert np.isfinite(out).all()


def test_registry_alias_still_resolves():
    assert get_metric("lid").func is lid


# ---- scale invariance: what the clamp broke ----

def test_lid_is_unchanged_by_a_7_6e_16_rescale():
    x = _subspace_cloud()
    k = 20
    # Relative allowance scales with float32 precision and the feature/neighbor
    # summation lengths; no absolute distance or dimension tolerance.
    operations = x.shape[1] * k
    eps = np.finfo(x.dtype).eps
    rtol = operations * eps / (1 - operations * eps)
    np.testing.assert_allclose(lid(x * TINY, k=k), lid(x, k=k), rtol=rtol, atol=0)


# ---- duplicates: the case the old comment named and did not handle ----

def test_dedup_is_an_exact_no_op_on_a_duplicate_free_cloud():
    """Callers whose embeddings never had duplicates must see no change."""
    x = _subspace_cloud()
    np.testing.assert_array_equal(
        lid(x, k=20, return_per_sample=True, dedup=True),
        lid(x, k=20, return_per_sample=True, dedup=False),
    )


def test_identical_rows_receive_identical_lid():
    """Duplicates are the same point, so they cannot carry different dimensions."""
    base = _subspace_cloud(n=300)
    x = np.concatenate([base, base[:20]])
    out = lid(x, k=20, return_per_sample=True)
    np.testing.assert_array_equal(out[:20], out[300:])


def test_a_duplicate_heavy_cloud_recovers_the_subspace_dimension():
    base = _subspace_cloud(n=400)
    x = np.concatenate([base, base])
    assert 3.0 < lid(x, k=20) < 8.0


# ---- refuse rather than return a number ----

def test_an_all_zero_embedding_raises():
    with pytest.raises(ValueError, match="no geometry"):
        lid(np.zeros((100, 8), dtype=np.float32), k=20)


def test_a_non_finite_embedding_raises():
    x = np.ones((100, 8), dtype=np.float32)
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="no geometry"):
        lid(x, k=20)


def test_too_few_distinct_points_raises():
    x = np.repeat(_subspace_cloud(n=15), 40, axis=0)
    with pytest.raises(ValueError, match="distinct points"):
        lid(x, k=20)
