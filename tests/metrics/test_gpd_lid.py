"""Tail-index (generalized LID) estimators on neighbour distances."""
import numpy as np
import pytest

from manylatents.metrics import compute_metric
from manylatents.metrics.gpd_lid import (
    GeneralizedLID,
    exponentiality,
    hill_tail_index,
    pickands_tail_index,
    tail_distances,
)
from manylatents.metrics.lid import LocalIntrinsicDimensionality
from manylatents.utils.exceptions import MeasurementUnavailable


def _power_law_distances(n_points, k, m, rng, n_total=2000):
    """k smallest of n_total radii whose CDF is r**m on [0, 1]."""
    u = np.sort(rng.random((n_points, n_total)), axis=1)[:, :k]
    return u ** (1.0 / m)


def _scaled_brute(reference, query, k, exclude=None):
    rms = np.sqrt(np.mean(reference.astype(np.float64) ** 2))
    d = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2) / rms
    if exclude is not None:
        d[np.arange(len(query)), exclude] = np.inf
    return np.sort(d, axis=1)[:, :k]


# --- estimators on synthetic distances ---------------------------------------

def test_hill_recovers_dimension():
    d = _power_law_distances(400, 100, m=5.0, rng=np.random.default_rng(0))
    xi = hill_tail_index(d)
    assert xi.shape == (400,)
    assert np.all(xi > 0)
    assert abs(np.mean(1.0 / xi) - 5.0) < 0.25


def test_pickands_recovers_inverse_dimension():
    d = _power_law_distances(2000, 100, m=5.0, rng=np.random.default_rng(1))
    assert abs(np.median(pickands_tail_index(d)) - 0.2) < 0.06


def test_pickands_is_negative_for_a_bounded_neighbourhood():
    # No neighbour closer than 1: inverse distances have a hard upper bound.
    rng = np.random.default_rng(2)
    u = np.sort(rng.random((500, 2000)), axis=1)[:, :100]
    assert np.median(pickands_tail_index(1.0 + u)) < -0.5


def test_exponentiality_high_for_power_law_low_otherwise():
    d = _power_law_distances(300, 200, m=4.0, rng=np.random.default_rng(3))
    assert np.mean(exponentiality(d)) > 0.6
    # log-ratios spread uniformly on [0, 1] instead of exponentially
    t = np.linspace(1.0, 0.0, 50)
    uniform_log = np.exp(-t)[None, :]
    assert exponentiality(uniform_log)[0] < 0.2


def test_estimators_reject_bad_distance_arrays():
    with pytest.raises(MeasurementUnavailable):
        hill_tail_index(np.array([[1.0]]))                      # k < 2
    with pytest.raises(MeasurementUnavailable):
        hill_tail_index(np.array([[0.0, 1.0, 2.0]]))            # zero distance
    with pytest.raises(MeasurementUnavailable):
        hill_tail_index(np.array([[2.0, 1.0, 3.0]]))            # not ascending
    with pytest.raises(MeasurementUnavailable):
        pickands_tail_index(np.array([[1.0, 2.0, 3.0]]))        # k < 4


def test_undefined_points_are_reported_not_filled():
    d = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])            # row 1: all radii equal
    with pytest.raises(MeasurementUnavailable) as err:
        hill_tail_index(d)
    assert err.value.indices.tolist() == [1]
    tied = np.array([[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]]) # k=8, j=2: s_2 == s_4
    with pytest.raises(MeasurementUnavailable) as err:
        pickands_tail_index(tied)
    assert err.value.indices.tolist() == [0]


# --- neighbour distances ------------------------------------------------------

def test_transductive_hill_equals_lid():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(250, 6))
    x = np.vstack([x, x[:3]])                                   # duplicate observations
    d = tail_distances(x, k=20)
    assert d.shape == (253, 20)
    lid = LocalIntrinsicDimensionality(x, k=20, return_per_sample=True)
    np.testing.assert_allclose(1.0 / hill_tail_index(d), lid, rtol=1e-9)


def test_reference_mode_matches_brute_force():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(500, 8))
    query = rng.normal(size=(50, 8))
    d = tail_distances(query, k=15, reference=reference)
    assert d.shape == (50, 15)
    assert np.all(d > 0) and np.all(np.diff(d, axis=1) >= 0)
    np.testing.assert_allclose(d, _scaled_brute(reference, query, 15), rtol=1e-3)


def test_reference_mode_excludes_a_coincident_reference_row():
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(400, 5))
    rows = np.array([0, 17, 399])
    d = tail_distances(reference[rows], k=10, reference=reference)
    assert np.all(d > 0)
    np.testing.assert_allclose(
        d, _scaled_brute(reference, reference[rows], 10, exclude=rows), rtol=1e-3
    )


def test_reference_duplicates_count_once():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(300, 4))
    query = rng.normal(size=(20, 4))
    doubled = np.vstack([reference, reference[:50]])
    np.testing.assert_allclose(
        hill_tail_index(tail_distances(query, k=12, reference=doubled)),
        hill_tail_index(tail_distances(query, k=12, reference=reference)),
        rtol=1e-3,
    )


def test_reference_mode_is_scale_free():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(300, 4))
    query = rng.normal(size=(20, 4))
    a = hill_tail_index(tail_distances(query, k=12, reference=reference))
    b = hill_tail_index(tail_distances(1000.0 * query, k=12, reference=1000.0 * reference))
    np.testing.assert_allclose(a, b, rtol=1e-3)


def test_reference_mode_needs_k_below_distinct_reference_size():
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(10, 3))
    with pytest.raises(MeasurementUnavailable):
        tail_distances(rng.normal(size=(2, 3)), k=10, reference=reference)


# --- registered metric --------------------------------------------------------

def test_metric_scalar_per_sample_and_registry():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(200, 5))
    per_point = GeneralizedLID(x, k=20, return_per_sample=True)
    assert per_point.shape == (200,)
    assert GeneralizedLID(x, k=20) == pytest.approx(float(np.mean(per_point)))
    assert compute_metric("gpd_lid", x, k=20) == pytest.approx(GeneralizedLID(x, k=20))
    signed = GeneralizedLID(x, k=40, method="pickands", return_per_sample=True)
    assert signed.shape == (200,) and np.all(np.isfinite(signed))
    with pytest.raises(MeasurementUnavailable):
        GeneralizedLID(x, method="moments")


def test_metric_with_reference_cloud():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(600, 5))
    query = rng.normal(size=(30, 5))
    xi = GeneralizedLID(query, k=20, reference=reference, return_per_sample=True)
    np.testing.assert_allclose(
        xi, hill_tail_index(tail_distances(query, k=20, reference=reference))
    )


@pytest.mark.parametrize("row", [[0., 2., 3., 4.], [2., 1., 3., 4.], [1., 2., 3., np.nan]])
def test_invalid_distance_rows_report_indices(row):
    with pytest.raises(MeasurementUnavailable) as err:
        hill_tail_index(np.array([[1., 2., 3., 4.], row]))
    assert err.value.indices.tolist() == [1]


def test_estimators_handle_extreme_finite_distance_scales():
    d = np.array([[1e-310, 2e-310, 3e-310, 4e-310]])
    np.testing.assert_allclose(pickands_tail_index(d), pickands_tail_index(np.array([[1., 2., 3., 4.]])))
    wide = np.array([[1e-300, 1e-100, 1e100, 1e300]])
    assert np.isfinite(hill_tail_index(wide)).all()
    assert np.isfinite(exponentiality(wide)).all()


def test_query_precision_collapse_is_not_treated_as_identity():
    reference = np.array([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    query = np.array([[1. + 1e-10, 0.], [0., 0.]])
    with pytest.raises(MeasurementUnavailable) as err:
        tail_distances(query, k=2, reference=reference)
    assert err.value.indices.tolist() == [0]
