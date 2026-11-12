"""Shared distance estimator preserves the metric's arithmetic."""
import numpy as np
import pytest
from manylatents.metrics import lid
from manylatents.utils.exceptions import MeasurementUnavailable


def test_extracted_formula_matches_existing_metric(monkeypatch):
    distances = np.array([[1., 2., 3.], [2., 3., 4.], [1., 3., 5.], [2., 4., 6.]])
    monkeypatch.setattr(lid, 'compute_knn', lambda *a, **kw: (distances, None))
    points = np.arange(8.).reshape(4, 2)
    expected = lid.LocalIntrinsicDimensionality(points, k=3, return_per_sample=True)
    _, inverse = lid._distinct_rows(points)
    np.testing.assert_array_equal(lid._lid_from_distances(distances)[inverse], expected)


@pytest.mark.parametrize('distances', [[[0., 1.]], [[1., 1.]], [[1., np.inf]], [[1., np.nan]]])
def test_bad_radii_raise(distances):
    with pytest.raises(MeasurementUnavailable):
        lid._lid_from_distances(np.array(distances))
