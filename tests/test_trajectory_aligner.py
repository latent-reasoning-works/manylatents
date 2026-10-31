"""Known-answer tests for TrajectoryAligner (the v0 same-model aligner).

Every assertion is a bound with an analytic answer — identity self-alignment,
prefix-truncation, resample-to-same-length, localized divergence — per the
"verifies-with" contract: an aligner that can't be pinned to a known bound
shouldn't ship.
"""
import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.trajectory_aligner import TrajectoryAligner


def _traj(s: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((s, d))


def test_is_latent_module():
    assert isinstance(TrajectoryAligner(), LatentModule)


# ---- identity bound: fit_transform(A, A) == A and residual(A, A) == 0 ----
@pytest.mark.parametrize("matcher", ["truncate", "resample"])
def test_identity_self_alignment(matcher):
    A = _traj(12, 8)
    al = TrajectoryAligner(matcher=matcher)
    np.testing.assert_allclose(al.fit_transform(A, A), A, atol=1e-12)
    np.testing.assert_allclose(al.residual(A, A), 0.0, atol=1e-12)


@pytest.mark.parametrize("matcher", ["truncate", "resample"])
def test_y_none_is_self_align(matcher):
    A = _traj(10, 5)
    al = TrajectoryAligner(matcher=matcher)
    np.testing.assert_allclose(al.fit_transform(A), A, atol=1e-12)  # y=None -> reference is self
    np.testing.assert_allclose(al.residual(A), 0.0, atol=1e-12)


# ---- truncate bound: a prefix aligns with zero residual on the overlap ----
def test_truncate_prefix_zero_residual():
    ref = _traj(20, 6)
    src = ref[:8].copy()
    al = TrajectoryAligner(matcher="truncate")
    al.fit(src, ref)
    out = al.transform(src)
    assert out.shape == (8, 6)  # min(8, 20)
    np.testing.assert_allclose(al.residual(src, ref), 0.0, atol=1e-12)


# ---- resample bound: to the same length is the identity; shape onto the ref grid ----
def test_resample_same_length_is_identity():
    A = _traj(17, 5)
    al = TrajectoryAligner(matcher="resample")
    al.fit(A, A)
    np.testing.assert_allclose(al.transform(A), A, atol=1e-9)
    np.testing.assert_allclose(al.residual(A, A), 0.0, atol=1e-12)


def test_resample_puts_source_on_reference_grid():
    src, ref = _traj(15, 4, 1), _traj(30, 4, 2)
    al = TrajectoryAligner(matcher="resample")
    al.fit(src, ref)
    assert al.transform(src).shape == (30, 4)


# ---- the research statistic: divergence localizes at the split step ----
def test_divergence_localizes_at_split():
    d = 6
    common = _traj(20, d, 3)
    clean = common.copy()
    misleading = common.copy()
    misleading[10:] = _traj(10, d, 99)  # twins agree for 10 steps, then split
    al = TrajectoryAligner(matcher="truncate")
    al.fit(misleading, clean)
    r = al.residual(misleading, clean)
    assert np.all(r[:10] < 1e-9)      # aligned before the split
    assert r[10:].mean() > 0.1        # divergent after


# ---- residual properties ----
def test_residual_nonnegative():
    r = TrajectoryAligner(matcher="truncate").residual(_traj(14, 7, 1), _traj(9, 7, 2))
    assert np.all(r >= -1e-12)


def test_transform_preserves_torch_type():
    A = torch.randn(10, 5)
    out = TrajectoryAligner(matcher="truncate").fit_transform(A, A)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, A)


# ---- v0 guards: cross-model mapper is a reserved (unimplemented) seam ----
def test_mapper_is_not_implemented_in_v0():
    with pytest.raises(NotImplementedError):
        TrajectoryAligner(mapper="procrustes")


def test_cross_model_dim_mismatch_is_rejected():
    al = TrajectoryAligner(matcher="truncate")
    with pytest.raises(ValueError):
        al.fit(_traj(10, 8), _traj(10, 4))  # different hidden dim -> needs the mapper


def test_bad_matcher_rejected():
    with pytest.raises(ValueError):
        TrajectoryAligner(matcher="nope")
