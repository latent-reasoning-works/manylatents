"""Search ceilings may exceed the cohort; requested neighborhoods may not."""
from types import SimpleNamespace

import numpy as np
import pytest

from manylatents.algorithms.latent.selective_correction import _compute_mismatch_labels
from manylatents.metrics.loglog_consistency import LogLogConsistency
from manylatents.metrics.mismatch_ratio import MismatchRatio, _compute_kstar
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.knn import compute_knn


@pytest.fixture
def data():
    return np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)


@pytest.mark.parametrize("ceiling", [200, 400])
@pytest.mark.parametrize("cached", [False, True])
def test_mismatch_sweep_is_bounded_by_available_neighbors(data, ceiling, cached):
    cache = {} if cached else None
    expected, expected_grid = _compute_kstar(data, k_max=199, cache=cache)
    actual, grid = _compute_kstar(data, k_max=ceiling, cache=cache)
    np.testing.assert_array_equal(grid, expected_grid)
    np.testing.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))
    assert 5 <= grid.min() <= grid.max() <= 199
    assert np.all(np.isin(actual, grid))

    # Bounding the diagnostic must not weaken a direct neighborhood request,
    # even after the diagnostic has populated the cache.
    with pytest.raises(MeasurementUnavailable, match="kNN"):
        compute_knn(data, k=ceiling, cache=cache)


def test_selective_correction_mismatch_diagnostic_at_default_ceiling(data):
    weights = np.ones((len(data), len(data))) - np.eye(len(data))
    module = SimpleNamespace(affinity=lambda **kwargs: weights)
    labels, ratios = _compute_mismatch_labels(data, module, k_max=len(data))
    result = MismatchRatio(
        data[:, :2], dataset=SimpleNamespace(data=data), module=module, k=len(data),
    )
    assert labels.shape == ratios.shape == (200,)
    assert labels.dtype == np.bool_
    assert np.all(np.isfinite(ratios))
    np.testing.assert_allclose(ratios, result["v"])
    np.testing.assert_array_equal(labels, (ratios > 1.0) | (ratios < 0.5))


@pytest.mark.parametrize("ceiling", [200, 400])
@pytest.mark.parametrize("cached", [False, True])
def test_loglog_sweep_is_bounded_by_available_neighbors(data, ceiling, cached):
    cache = {} if cached else None
    expected = LogLogConsistency(data, k=199, cache=cache)
    actual = LogLogConsistency(data, k=ceiling, cache=cache)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])
        assert np.all(np.isfinite(actual[key]))
    assert actual["k_values"].max() <= 199


@pytest.mark.parametrize("diagnostic", ["mismatch", "loglog"])
@pytest.mark.parametrize("n_samples,ceiling,steps", [
    (200, 4, 20),   # ceiling below k_min
    (200, 5, 20),   # only one distinct k
    (200, 200, 1),  # too few requested steps
    (200, 200, 0),  # empty grid
    (200, 0, 20),   # no positive k
    (200, -1, 20),
    (5, 200, 20),   # data cannot support k_min
    (1, 200, 20),   # no other neighbors
    (0, 200, 20),
])
def test_unusable_sweep_is_unavailable(diagnostic, n_samples, ceiling, steps):
    data = np.random.default_rng(0).normal(size=(n_samples, 4))
    with pytest.raises(MeasurementUnavailable, match="k sweep"):
        if diagnostic == "mismatch":
            _compute_kstar(data, k_max=ceiling, k_steps=steps)
        else:
            LogLogConsistency(data, k=ceiling, k_steps=steps)


def test_mismatch_requires_enough_sweep_points_to_search(data):
    with pytest.raises(MeasurementUnavailable, match="at least 3 distinct k"):
        _compute_kstar(data, k_max=6)
