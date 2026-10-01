"""compute_knn_query: neighbours of query rows searched in a separate reference set."""
import numpy as np
import pytest

from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.knn import compute_knn_query


def _brute(reference, query, k):
    d = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, idx, axis=1), idx


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(300, 6))
    query = rng.normal(size=(40, 6))
    distances, indices = compute_knn_query(reference, query, k=7)
    expected_d, expected_i = _brute(reference, query, 7)
    assert distances.shape == indices.shape == (40, 7)
    np.testing.assert_array_equal(indices, expected_i)
    np.testing.assert_allclose(distances, expected_d, rtol=1e-4, atol=1e-5)
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_coincident_query_returns_that_reference_row_first():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(100, 5))
    distances, indices = compute_knn_query(reference, reference[[3, 42]], k=4)
    assert indices[:, 0].tolist() == [3, 42]
    assert distances[:, 0] == pytest.approx(0.0, abs=1e-3)


def test_k_may_equal_reference_size():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(5, 3))
    distances, _ = compute_knn_query(reference, rng.normal(size=(2, 3)), k=5)
    assert distances.shape == (2, 5)


@pytest.mark.parametrize("k", [0, 6, 2.0, True])
def test_rejects_invalid_k(k):
    rng = np.random.default_rng(3)
    with pytest.raises(MeasurementUnavailable):
        compute_knn_query(rng.normal(size=(5, 3)), rng.normal(size=(2, 3)), k=k)


def test_rejects_dimension_mismatch_and_non_finite():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(20, 3))
    with pytest.raises(MeasurementUnavailable):
        compute_knn_query(reference, rng.normal(size=(2, 4)), k=2)
    bad = rng.normal(size=(2, 3))
    bad[0, 0] = np.nan
    with pytest.raises(MeasurementUnavailable):
        compute_knn_query(reference, bad, k=2)
