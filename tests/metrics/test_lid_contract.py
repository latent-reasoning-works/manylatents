"""Regression cases for LID's measurement domain, independent of its formula."""

import importlib
import sys

import numpy as np
import pytest

from manylatents.metrics.lid import LocalIntrinsicDimensionality as lid
from manylatents.utils.exceptions import MeasurementUnavailable


@pytest.fixture(params=["sklearn", "faiss"])
def backend(request, monkeypatch):
    if request.param == "sklearn":
        monkeypatch.setitem(sys.modules, "faiss", None)
    else:
        faiss = pytest.importorskip("faiss")
        # Exercise the real CPU implementation even on machines with a GPU.
        monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0)
        import sklearn.neighbors

        def unexpected_fallback(*args, **kwargs):
            pytest.fail("FAISS coverage must not silently fall back to sklearn")

        monkeypatch.setattr(sklearn.neighbors, "NearestNeighbors", unexpected_fallback)


@pytest.mark.parametrize("k", [0, 1, -1, True, np.bool_(False), 2.5, "2", np.nan])
def test_invalid_requested_k_is_unavailable(k):
    with pytest.raises(MeasurementUnavailable, match="k.*integer.*2"):
        lid(np.arange(30).reshape(10, 3), k=k)


def test_absent_k_resolves_to_lid_default():
    x = np.random.default_rng(8).normal(size=(40, 4))
    assert lid(x, k=None) == lid(x)


def test_numpy_integer_k_is_supported():
    x = np.random.default_rng(8).normal(size=(40, 4))
    assert lid(x, k=np.int64(2)) == lid(x, k=2)


@pytest.mark.parametrize("per_sample", [False, True])
def test_equal_neighbor_radii_are_unavailable(backend, per_sample):
    with pytest.raises(MeasurementUnavailable, match="log.*sum"):
        lid(np.eye(4), k=2, return_per_sample=per_sample)


def test_float32_conversion_cannot_merge_distinct_observations(backend):
    x = np.random.default_rng(0).normal(size=(40, 4))
    unresolved = np.concatenate([x, x + 1e-12])
    assert np.unique(unresolved, axis=0).shape[0] == 80
    with pytest.raises(MeasurementUnavailable, match="distinct.*precision"):
        lid(unresolved, k=20)


def test_disabled_dedup_rejects_duplicates_before_knn(backend, monkeypatch):
    x = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    x = np.concatenate([x, x[:1]])
    def unexpected_knn(*args, **kwargs):
        pytest.fail("duplicate rejection must precede backend round-off")
    monkeypatch.setattr(importlib.import_module("manylatents.metrics.lid"), "compute_knn", unexpected_knn)
    with pytest.raises(MeasurementUnavailable, match="duplicat"):
        lid(x, k=20, dedup=False)


@pytest.mark.parametrize("x", [np.zeros((30, 2)), np.full((30, 2), np.inf), np.full((30, 2), np.nan)])
def test_no_geometry_uses_shared_unavailable_type(x):
    with pytest.raises(MeasurementUnavailable, match="no geometry"):
        lid(x)


def test_requested_k_is_not_clamped_to_distinct_population(backend):
    x = np.repeat(np.arange(4)[:, None], 10, axis=0)
    with pytest.raises(MeasurementUnavailable, match="distinct points"):
        lid(x, k=4)


@pytest.mark.parametrize("bad_distance", [0.0, -1.0, np.inf, np.nan])
def test_invalid_backend_distances_are_unavailable(monkeypatch, bad_distance):
    module = importlib.import_module("manylatents.metrics.lid")
    distances = np.tile([1.0, 2.0], (4, 1))
    distances[0, 0] = bad_distance
    monkeypatch.setattr(module, "compute_knn", lambda *args, **kwargs: (distances, None))
    with pytest.raises(MeasurementUnavailable, match="distance"):
        lid(np.arange(4)[:, None], k=2)


def test_one_unavailable_row_refuses_the_whole_estimate(monkeypatch):
    module = importlib.import_module("manylatents.metrics.lid")
    distances = np.tile([1.0, 2.0], (4, 1))
    distances[0] = 2.0
    monkeypatch.setattr(module, "compute_knn", lambda *args, **kwargs: (distances, None))
    with pytest.raises(MeasurementUnavailable, match="log.*sum"):
        lid(np.arange(4)[:, None], k=2)


def test_genuine_duplicates_keep_sample_identity_and_multiplicity(backend):
    x = np.random.default_rng(8).normal(size=(40, 4))
    repeated = np.concatenate([x, x[:3], x[:3]])
    values = lid(repeated, k=5, return_per_sample=True)
    np.testing.assert_array_equal(values[:3], values[40:43])
    np.testing.assert_array_equal(values[:3], values[43:])
    assert lid(repeated, k=5) == np.mean(values)


def test_nonzero_subunit_dimension_is_not_arbitrarily_refused(backend):
    # On a line, clustered pairs with a distant third point can legitimately
    # yield a finite-sample estimate below one; ambient dimension is no bound.
    assert 0 < lid(np.array([[0.0], [1.0], [100.0], [101.0]]), k=2) < 1


@pytest.mark.parametrize("x", [np.arange(30), np.empty((0, 3)), np.empty((30, 0)), np.ones((30, 2), dtype=complex) * (1 + 2j)])
def test_embedding_requires_nonempty_real_matrix(x):
    with pytest.raises(MeasurementUnavailable, match="nonempty.*real.*matrix"):
        lid(x, k=2)


def test_log_ratio_underflow_does_not_become_zero_dimension(monkeypatch):
    module = importlib.import_module("manylatents.metrics.lid")
    distances = np.tile(np.array([np.finfo(np.float32).tiny, 1e10], dtype=np.float32), (4, 1))
    monkeypatch.setattr(module, "compute_knn", lambda *args, **kwargs: (distances, None))
    assert lid(np.arange(4)[:, None], k=2) > 0


def test_scale_invariance_on_each_backend(backend):
    x = np.random.default_rng(8).normal(size=(80, 4)).astype(np.float32)
    k = 5
    operations = x.shape[1] * k
    eps = np.finfo(x.dtype).eps
    rtol = operations * eps / (1 - operations * eps)
    np.testing.assert_allclose(lid(x * 7.6e-16, k=k), lid(x, k=k), rtol=rtol, atol=0)
