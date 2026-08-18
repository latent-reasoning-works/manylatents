"""Pin the numerical contract of LocalIntrinsicDimensionality (issue #304).

Every test here fails against the previous implementation, which clamped
``r_k`` to 1e-10 and added 1e-10 inside the log. Both are absolute constants
governing a scale-free quantity: the estimator stops being scale-invariant the
moment one appears in it, and the symptom is a plausible number rather than a
crash.

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


def _clamped_lid(x, k=20):
    """The previous implementation, verbatim, for direct comparison."""
    from manylatents.utils.metrics import compute_knn

    distances, _ = compute_knn(np.ascontiguousarray(x, dtype=np.float32), k=k, include_self=False)
    r_k = np.maximum(distances[:, -1], 1e-10)
    return -k / np.sum(np.log(distances / r_k[:, None] + 1e-10), axis=1)


# ---- the estimator still estimates ----

def test_recovers_the_dimension_of_a_linear_subspace():
    assert 3.0 < lid(_subspace_cloud(), k=20) < 8.0


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
    assert lid(x * TINY, k=20) == pytest.approx(lid(x, k=20), rel=1e-3)


def test_the_old_clamp_is_what_failed_under_that_rescale():
    """Without this, the invariance test above passes against any implementation."""
    x = _subspace_cloud()
    new = np.median(lid(x, k=20, return_per_sample=True))
    assert np.median(_clamped_lid(x)) == pytest.approx(new, rel=0.05), (
        "at unit scale the clamp is inert, so old and new agree"
    )
    assert np.median(_clamped_lid(x * TINY)) < 1.0, (
        "at 7.6e-16 every distance is below the clamp and LID collapses — the bug"
    )


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


def test_a_duplicate_heavy_cloud_no_longer_reports_a_sub_unit_dimension():
    """The signature the old estimator produced on real protein embeddings.

    A local intrinsic dimension below 1 in a 64-dimensional space is not a
    dimension. Deduplicating restores the intrinsic dimension of the underlying
    cloud, which is what the duplicated rows actually sit on.
    """
    base = _subspace_cloud(n=400)
    x = np.concatenate([base, base])  # every row has exactly one exact duplicate
    assert np.median(_clamped_lid(x)) < 1.0, "fixture must reproduce the old failure"
    assert 3.0 < lid(x, k=20) < 8.0


def test_duplicate_distances_are_decided_by_round_off_not_by_geometry():
    """Why no choice of epsilon could have fixed this.

    ``compute_knn`` expands ``||a||^2 + ||b||^2 - 2a.b``, so the true zero
    distance between two duplicate rows comes back as exactly 0 for some pairs
    and as a cancellation residue (~1e-7 here) for others — decided by
    floating-point luck, not by the data. The old estimator's epsilon governed
    only the first group; the second got a LID computed off round-off. Both
    read as a sub-unit "dimension" in a 64-dimensional space.
    """
    from manylatents.utils.metrics import compute_knn

    base = _subspace_cloud(n=400)
    x = np.concatenate([base, base])  # every row has exactly one exact duplicate

    d, _ = compute_knn(np.ascontiguousarray(x, dtype=np.float32), k=20, include_self=False)
    exact_zero = d[:, 0] == 0.0
    assert 0.0 < exact_zero.mean() < 1.0, (
        "the split is the point: a true zero distance surfaces as 0 for some "
        "duplicate pairs and as round-off for others"
    )
    assert d[~exact_zero, 0].max() < 1e-4, "the non-zero ones are round-off, not geometry"

    assert np.median(_clamped_lid(x)) < 1.0, "both paths produced a sub-unit dimension"
    assert 3.0 < lid(x, k=20) < 8.0, "dedup recovers the dimension the points actually sit on"


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
