"""Per-point scores of query points against a separate reference cloud."""
import numpy as np
import pytest

from manylatents.metrics.reference_geometry import (
    knn_distance_score,
    local_participation_ratio,
    lof_novelty_score,
    pca_reference_scores,
    standardize_against,
)
from manylatents.utils.exceptions import MeasurementUnavailable


def test_knn_distance_score_matches_brute_force():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 4))
    query = rng.normal(size=(15, 4))
    d = np.sort(np.linalg.norm(query[:, None] - reference[None], axis=2), axis=1)[:, :5]
    np.testing.assert_allclose(
        knn_distance_score(query, reference, k=5), d.mean(axis=1), rtol=1e-4
    )


def test_outlier_scores_exceed_inlier_scores():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(500, 3))
    query = np.vstack([np.zeros((1, 3)), np.full((1, 3), 12.0)])   # inlier, outlier
    for score in (knn_distance_score, lof_novelty_score):
        inlier, outlier = score(query, reference, k=20)
        assert outlier > inlier
    assert lof_novelty_score(query, reference, k=20)[0] == pytest.approx(1.0, abs=0.3)


def test_pca_scores_split_in_plane_and_off_plane_parts():
    rng = np.random.default_rng(2)
    # reference lies in the first two coordinates of R^5, centred at the origin
    plane = rng.normal(size=(400, 2))
    reference = np.hstack([plane - plane.mean(axis=0), np.zeros((400, 3))])
    query = np.array([[3.0, 4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 2.0]])
    scores = pca_reference_scores(query, reference, n_components=2)
    np.testing.assert_allclose(scores["leading_norm"], [5.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(scores["residual_norm"], [0.0, 2.0], atol=1e-8)


def test_pca_handles_more_features_than_reference_points():
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(30, 500))
    query = rng.normal(size=(4, 500))
    scores = pca_reference_scores(query, reference, n_components=10)
    total = np.linalg.norm(query - reference.mean(axis=0), axis=1)
    np.testing.assert_allclose(
        scores["leading_norm"] ** 2 + scores["residual_norm"] ** 2, total ** 2, rtol=1e-8
    )


def test_pca_rejects_more_components_than_rank():
    rng = np.random.default_rng(4)
    plane = np.hstack([rng.normal(size=(50, 2)), np.zeros((50, 3))])
    with pytest.raises(MeasurementUnavailable):
        pca_reference_scores(plane[:2], plane, n_components=3)


def test_local_participation_ratio_tracks_local_dimension():
    rng = np.random.default_rng(5)
    plane = np.hstack([rng.normal(size=(600, 2)), np.zeros((600, 8))])
    pr_plane = local_participation_ratio(plane[:40] + 0.0, plane[40:], k=30)
    assert pr_plane.shape == (40,)
    assert 1.2 < np.median(pr_plane) <= 2.0 + 1e-6
    full = rng.normal(size=(600, 10))
    assert np.median(local_participation_ratio(full[:40], full[40:], k=50)) > 5.0


def test_local_participation_ratio_is_chunk_invariant():
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(300, 6))
    query = rng.normal(size=(25, 6))
    np.testing.assert_allclose(
        local_participation_ratio(query, reference, k=15, chunk_size=4),
        local_participation_ratio(query, reference, k=15, chunk_size=256),
    )


def test_standardize_against_uses_reference_moments():
    rng = np.random.default_rng(7)
    reference = rng.normal(loc=3.0, scale=2.0, size=(1000, 4))
    query = rng.normal(size=(10, 4))
    query_z, reference_z, kept = standardize_against(query, reference)
    assert kept.all()
    np.testing.assert_allclose(reference_z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(reference_z.std(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        query_z, (query - reference.mean(axis=0)) / reference.std(axis=0)
    )


def test_standardize_against_constant_columns():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(100, 3))
    reference[:, 1] = 7.0
    query = rng.normal(size=(5, 3))
    with pytest.raises(MeasurementUnavailable):
        standardize_against(query, reference)
    query_z, reference_z, kept = standardize_against(query, reference, constant="drop")
    assert kept.tolist() == [True, False, True]
    assert query_z.shape == (5, 2) and reference_z.shape == (100, 2)


def test_lof_matches_sklearn_on_resolvable_reference():
    from sklearn.neighbors import LocalOutlierFactor
    rng = np.random.default_rng(50)
    reference, query = rng.normal(size=(100, 4)), rng.normal(size=(12, 4))
    expected = -LocalOutlierFactor(n_neighbors=8, novelty=True).fit(reference).score_samples(query)
    np.testing.assert_allclose(lof_novelty_score(query, reference, k=8), expected, rtol=1e-5)


def test_lof_refuses_zero_reachability_instead_of_epsilon():
    with pytest.raises(MeasurementUnavailable) as err:
        lof_novelty_score(np.array([[0., 0.], [2., 2.]]), np.zeros((10, 2)), k=3)
    assert err.value.indices.tolist() == [0, 1]


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_local_participation_ratio_refuses_invalid_chunk_size(chunk_size):
    rng = np.random.default_rng(51)
    with pytest.raises(MeasurementUnavailable):
        local_participation_ratio(rng.normal(size=(4, 3)), rng.normal(size=(20, 3)), k=5, chunk_size=chunk_size)


def test_standardization_refuses_overflowed_moments():
    with pytest.raises(MeasurementUnavailable):
        standardize_against(np.array([[1e200]]), np.array([[-1e200], [1e200]]))


def test_pca_refuses_nonfinite_scores_with_rows():
    with pytest.raises(MeasurementUnavailable) as err:
        pca_reference_scores(np.array([[1., 1.], [1e200, 1e200]]), np.array([[-1., 0.], [1., 0.]]), 1)
    assert err.value.indices.tolist() == [1]


def test_decimal_constant_columns_are_recognized_exactly():
    reference = np.column_stack([np.arange(100.), np.full(100, 0.1)])
    with pytest.raises(MeasurementUnavailable):
        standardize_against(reference[:2], reference)
    _, _, kept = standardize_against(reference[:2], reference, constant="drop")
    assert kept.tolist() == [True, False]


def test_decimal_constant_cloud_has_no_pca_rank_or_local_spread():
    reference = np.full((100, 2), 0.1)
    with pytest.raises(MeasurementUnavailable):
        pca_reference_scores(reference[:2], reference, n_components=1)
    with pytest.raises(MeasurementUnavailable) as err:
        local_participation_ratio(reference[:2], reference, k=20)
    assert err.value.indices.tolist() == [0, 1]


# --- Mahalanobis distance to the reference cloud ---------------------------------


def _mahalanobis_direct(query, reference, ridge):
    mean = reference.mean(axis=0)
    covariance = np.cov(reference - mean, rowvar=False)
    covariance = covariance + ridge * np.trace(covariance) / covariance.shape[0] * np.eye(covariance.shape[0])
    centred = query - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centred, np.linalg.inv(covariance), centred))


def test_mahalanobis_matches_the_direct_formula():
    from manylatents.metrics.reference_geometry import mahalanobis_score

    rng = np.random.default_rng(20)
    reference = rng.normal(size=(400, 6)) @ rng.normal(size=(6, 6))
    query = rng.normal(size=(25, 6)) * 3.0
    for ridge in (1e-3, 0.1):
        np.testing.assert_allclose(
            mahalanobis_score(query, reference, ridge=ridge),
            _mahalanobis_direct(query, reference, ridge), rtol=1e-8,
        )


def test_mahalanobis_weights_directions_by_reference_spread():
    from manylatents.metrics.reference_geometry import mahalanobis_score

    rng = np.random.default_rng(21)
    reference = rng.normal(size=(2000, 2)) * np.array([10.0, 0.1])
    along_wide, along_narrow = np.array([[5.0, 0.0]]), np.array([[0.0, 5.0]])
    # the same Euclidean length is ordinary along the wide axis, extreme along the narrow one
    assert mahalanobis_score(along_narrow, reference)[0] > 20 * mahalanobis_score(along_wide, reference)[0]


def test_mahalanobis_is_invariant_to_a_common_rescaling():
    from manylatents.metrics.reference_geometry import mahalanobis_score

    rng = np.random.default_rng(22)
    reference = rng.normal(size=(300, 5))
    query = rng.normal(size=(10, 5))
    np.testing.assert_allclose(
        mahalanobis_score(query, reference),
        mahalanobis_score(1000.0 * query, 1000.0 * reference), rtol=1e-8,
    )


def test_mahalanobis_with_more_features_than_reference_points():
    from manylatents.metrics.reference_geometry import mahalanobis_score

    rng = np.random.default_rng(23)
    reference = rng.normal(size=(40, 300))
    query = rng.normal(size=(6, 300))
    scores = mahalanobis_score(query, reference, ridge=1e-2)
    assert scores.shape == (6,) and np.all(np.isfinite(scores)) and np.all(scores > 0)
    np.testing.assert_allclose(scores, _mahalanobis_direct(query, reference, 1e-2), rtol=1e-6)


@pytest.mark.parametrize("ridge", [0.0, -1.0, float("nan"), True])
def test_mahalanobis_rejects_a_non_positive_ridge(ridge):
    from manylatents.metrics.reference_geometry import mahalanobis_score

    rng = np.random.default_rng(24)
    with pytest.raises(MeasurementUnavailable):
        mahalanobis_score(rng.normal(size=(3, 4)), rng.normal(size=(50, 4)), ridge=ridge)


def test_mahalanobis_refuses_a_reference_without_spread():
    from manylatents.metrics.reference_geometry import mahalanobis_score

    with pytest.raises(MeasurementUnavailable):
        mahalanobis_score(np.ones((2, 3)), np.ones((10, 3)))
