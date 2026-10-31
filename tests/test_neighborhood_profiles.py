"""Deterministic fixtures for raw geometric objects, with no fitted layouts."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from manylatents.utils.neighborhood_profiles import (
    neighborhood_profiles, patch_moments, transported_patch_radii,
    influence_profiles,
)

X = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0]], dtype=np.float64)
RADII = np.array([.5, 1, 2], dtype=np.float64)
TOL = dict(atol=1e-10, rtol=1e-9)


def test_exact_duplicate_boundary_and_gaussian_fixture():
    result = neighborhood_profiles(X, X[[0, 0]], [0, None], RADII)
    assert_array_equal(result['hard_count'], [[1, 3, 4], [2, 4, 5]])
    assert_array_equal(result['ambient_sorted_ids'][0], [1, 2, 3, 4])
    expected = 1 + 2 * np.exp(-1 / RADII**2) + np.exp(-4 / RADII**2)
    assert_allclose(result['gaussian_mass'], [expected, expected + 1], **TOL)
    assert_allclose(result['normalized_mass'], [expected / 4, (expected + 1) / 5], **TOL)
    moments = result['moments']
    assert_allclose(moments['mean'][:, 2], [[0, .5, 0], [0, .4, 0]], **TOL)
    expected_cov = [[np.diag([2/3, 0, 0]), np.diag([.5, .75, 0])],
                    [np.diag([.5, 0, 0]), np.diag([.4, .64, 0])]]
    assert_allclose(moments['centered_moment'][:, 1:], expected_cov, **TOL)
    assert_allclose(result['mass_secants'], np.diff(np.log(result['gaussian_mass']), axis=1) / np.diff(np.log(RADII)), **TOL)
    assert_allclose(result['secant_radii'], np.sqrt(RADII[:-1]*RADII[1:]), **TOL)
    assert np.all(moments['named_q_status'][:, 1:] == 'LOW_SUPPORT')
    assert_array_equal(X, [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 2, 0]])


def test_empty_singleton_and_underflow_are_explicit():
    result = neighborhood_profiles(X, np.array([[1e5, 0, 0.]]), [None], RADII)
    assert np.all(result['hard_count'] == 0)
    assert np.all(result['moments']['covariance_status'] == 'EMPTY')
    assert np.isnan(result['moments']['centered_moment']).all()
    assert np.all(result['mass_status'] == 'KERNEL_UNDERFLOW')
    assert np.all(result['mass_secant_status'] == 'KERNEL_UNDERFLOW')
    assert np.all(result['count_secant_status'] == 'ZERO_COUNT')
    result = patch_moments(X, [[2]])
    assert_array_equal(result['centered_moment'][0], np.zeros((3, 3)))
    assert result['unbiased_status'][0] == 'INSUFFICIENT_SUPPORT'
    assert np.isnan(result['unbiased_covariance']).all()
    assert result['named_q_status'][0] == 'DEGENERATE_TANGENT_SPREAD'
    result = neighborhood_profiles(X[:1], X[:1], [0], RADII)
    assert np.all(result['mass_status'] == 'EMPTY_POPULATION')
    assert np.isnan(result['normalized_mass']).all()


def test_chart_windows_leave_ambient_kernel_unchanged():
    chart = X[:, :1].copy()
    ambient = neighborhood_profiles(X, X[[0]], [0], RADII)
    selected = neighborhood_profiles(X, X[[0]], [0], RADII,
                                     selection_coordinates=chart, selection_queries=chart[[0]])
    assert_array_equal(selected['hard_count'], [[2, 4, 4]])
    assert_allclose(selected['gaussian_mass'], ambient['gaussian_mass'], **TOL)
    assert selected['metadata']['window'] == 'selection_coordinates'
    assert_allclose(selected['moments']['mean'][0, 0], [0, 1, 0], **TOL)


@pytest.mark.parametrize('scale', [.5, 2.])
def test_similarity_rotation_translation_and_named_frame(scale):
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([8., -4, 2])
    radii = np.array([.4, 1.2, 2.2])  # No ties at tested boundaries.
    frame = np.eye(3)
    reference = neighborhood_profiles(X, X[[0, 2]], [0, 2], radii, frame=frame)
    transformed = X @ rotation * scale + shift
    result = neighborhood_profiles(transformed, transformed[[0, 2]], [0, 2], radii * scale,
                                   frame=rotation.T)
    assert_array_equal(result['hard_count'], reference['hard_count'])
    assert_allclose(result['gaussian_mass'], reference['gaussian_mass'], **TOL)
    assert_allclose(result['moments']['mean'], reference['moments']['mean'] @ rotation * scale + shift, **TOL)
    expected = np.swapaxes(rotation, 0, 1) @ reference['moments']['centered_moment'] @ rotation * scale**2
    assert_allclose(result['moments']['centered_moment'], expected, **TOL)
    assert_allclose(result['moments']['frame_moment'], reference['moments']['frame_moment'] * scale**2, **TOL)
    assert_allclose(result['moments']['named_q'], reference['moments']['named_q'], **TOL)


def test_high_dimensional_isometry_and_blocking():
    from scipy.fft import dct
    basis = dct(np.eye(50), norm='ortho')[:, :3]
    assert_allclose(basis.T @ basis, np.eye(3), **TOL)
    radii = np.array([.4, 1.2, 2.2])
    reference = neighborhood_profiles(X, X[[0, 2]], [0, 2], radii)
    mapped = X @ basis.T
    results = [neighborhood_profiles(mapped, mapped[[0, 2]], [0, 2], radii, frame=basis,
                                     query_block_size=block) for block in [1, 64]]
    for result in results:
        assert_array_equal(result['hard_count'], reference['hard_count'])
        assert_allclose(result['gaussian_mass'], reference['gaussian_mass'], **TOL)
        assert_allclose(result['moments']['frame_moment'], reference['moments']['centered_moment'], **TOL)
        assert_allclose(result['moments']['centered_moment'], basis @ reference['moments']['centered_moment'] @ basis.T, **TOL)
    assert_array_equal(results[0]['gaussian_mass'], results[1]['gaussian_mass'])


def test_permutation_remaps_identity_without_removing_duplicates():
    permutation = np.array([3, 0, 4, 2, 1])
    original = neighborhood_profiles(X, X[[0]], [0], RADII)
    permuted = neighborhood_profiles(X[permutation], X[[0]], [1], RADII)
    assert_array_equal(original['hard_count'], permuted['hard_count'])
    assert_allclose(original['gaussian_mass'], permuted['gaussian_mass'], **TOL)
    assert_allclose(original['moments']['centered_moment'], permuted['moments']['centered_moment'], **TOL)
    # Tie ordering follows current row IDs; geometric membership remains equal.
    for offset in original['inclusion_offsets'][0]:
        assert set(original['selection_sorted_ids'][0][:offset]) == set(permutation[permuted['selection_sorted_ids'][0][:offset]])


def test_full_input_hash_detects_interior_mutation():
    other = X.copy()
    other[2] = [10., 0, 0]
    assert_array_equal(X[[0, -1]], other[[0, -1]])
    from manylatents.utils.knn import _content_key
    assert _content_key(X) != _content_key(other)  # full-content key since v0.1.9 (#323): no partial-cache collision
    first = neighborhood_profiles(X, X[[0]], [0], RADII)
    second = neighborhood_profiles(other, X[[0]], [0], RADII)
    assert first['metadata']['X_sha256'] != second['metadata']['X_sha256']
    assert not np.array_equal(first['hard_count'], second['hard_count'])


def test_transport_ties_intruders_empty_and_strict_gap():
    result = transported_patch_radii(X, [0, 0, 0, 0, 0], [[1], [1, 2], [4], [], [1, 2, 3, 4]])
    assert_array_equal(result['status'], ['OK', 'BOUNDARY_TIE', 'INTERLEAVED', 'EMPTY_PATCH', 'EMPTY_COMPLEMENT'])
    assert_array_equal(result['strictly_selectable'], [True, False, False, None, True])
    assert_allclose(result['gap'][:3], [1, 0, -2], **TOL)
    assert np.isnan(result['inner_radius'][3])
    assert np.isnan(result['outer_radius'][4])
    scaled = transported_patch_radii(X * 2, [0], [[1]])
    assert_allclose(scaled['gap'], result['gap'][:1] * 2, **TOL)


def test_influence_normalization_self_mass_and_zero_rows():
    weights = np.array([[2., 1, 1], [0, 0, 0], [0, 1e-300, 1e-300]])
    distances = np.array([[0., 1, 2], [1, 0, 2], [2, 1, 0]])
    mask = np.array([[True, True, False]] * 3)
    result = influence_profiles(weights, distances, np.array([.5, 1., 2.]),
                                patch_mask=mask, self_indices=[0, 1, None])
    assert_allclose(result['row_sum'], [4, 0, 2e-300], atol=0, rtol=1e-12)
    assert_allclose(result['radial_cdf'][0], [.5, .75, 1], **TOL)
    assert_allclose(result['participation'][[0, 2]], [8/3, 2], **TOL)
    assert_allclose(result['patch_leakage'][[0, 2]], [.25, .5], **TOL)
    assert result['self_mass'][0] == 2
    assert result['self_fraction'][0] == .5
    assert result['status'][1] == 'ZERO_ROW'
    assert result['self_status'][2] == 'EXTERNAL_QUERY'
    assert np.isnan(result['radial_cdf'][1]).all()
    assert np.isnan(result['participation'][1])
    assert np.isnan(result['patch_leakage'][1])
    assert_allclose(result['distribution'][[0, 2]].sum(axis=1), 1, **TOL)
    assert_allclose(result['distribution'][0] * result['row_sum'][0], weights[0], **TOL)


@pytest.mark.parametrize('changes', [
    {'X': X.astype(np.float32)}, {'queries': np.full((1, 3), np.nan)},
    {'query_ids': [True]}, {'query_ids': [9]}, {'query_ids': [2]},
    {'radii': np.array([1., 1.])}, {'radii': np.array([0., 1.])},
    {'radii': np.array([1., np.inf])}, {'query_block_size': 0},
    {'frame': np.ones((3, 2))}, {'selection_coordinates': X[:, :1]},
])
def test_invalid_profile_inputs(changes):
    kwargs = dict(X=X, queries=X[[0]], query_ids=[0], radii=RADII)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        neighborhood_profiles(**kwargs)


def test_invalid_patch_and_operator_inputs():
    with pytest.raises(ValueError):
        patch_moments(X, [[1, 1]])
    with pytest.raises(ValueError):
        transported_patch_radii(X, [0], [[0, 1]])
    with pytest.raises(ValueError):
        influence_profiles(np.array([[-1., 1]]), np.array([[0., 1]]), RADII)
    with pytest.raises(ValueError):
        influence_profiles(np.array([[1., 1]]), np.array([[0., 1]]), RADII, patch_mask=np.array([[1, 0]]))


def test_exact_count_patch_differs_from_closed_ball_under_ties():
    profile = neighborhood_profiles(X, X[[0]], [0], RADII)
    exact_ids = profile['ambient_sorted_ids'][0][:2]
    exact = patch_moments(X, [exact_ids])
    assert_array_equal(exact_ids, [1, 2])
    assert exact['count'][0] == 2
    assert profile['hard_count'][0, 1] == 3
    assert_allclose(exact['mean'][0], [.5, 0, 0], **TOL)
    assert_allclose(exact['centered_moment'][0], np.diag([.25, 0, 0]), **TOL)


def test_unbiased_covariance_perquery_frame_and_stage_timings():
    frames = np.array([np.eye(3), np.eye(3)])
    profile = neighborhood_profiles(X, X[[0, 0]], [0, None], RADII, frame=frames)
    moments = profile['moments']
    assert_allclose(moments['unbiased_covariance'][0, 2], np.diag([2/3, 1., 0]), **TOL)
    assert_allclose(moments['frame_moment'], moments['centered_moment'], **TOL)
    for stage in ('distances_sorting_counts', 'mass', 'moments'):
        for clock in ('wall_seconds', 'process_seconds'):
            assert np.isfinite(profile['timings'][stage][clock])
            assert profile['timings'][stage][clock] >= 0


def test_extreme_finite_distances_do_not_underflow_from_squaring():
    # Moments may underflow at this scale, but radii/counts/mass remain meaningful.
    points = np.array([[0., 0., 0.], [1e-200, 0., 0.]])
    profile = neighborhood_profiles(points, points[[0]], [0], np.array([.5e-200, 2e-200]))
    assert_array_equal(profile['hard_count'], [[0, 1]])
    assert_allclose(profile['ambient_sorted_distances'][0], [1e-200], atol=0, rtol=1e-12)
    assert_allclose(profile['gaussian_mass'], [[np.exp(-4), np.exp(-.25)]], **TOL)


def test_adjacent_radii_with_equal_rounded_logs_have_finite_secants():
    from manylatents.utils.neighborhood_profiles import _secants
    radii = np.array([1e100, np.nextafter(1e100, np.inf)])
    assert np.log(radii[0]) == np.log(radii[1])
    secants, status = _secants(np.array([[1., 2.]]), radii, 'ZERO')
    assert status[0, 0] == 'OK'
    assert np.isfinite(secants).all()


def test_end_to_end_equal_rounded_log_radii_secants():
    radii = np.array([1e100, np.nextafter(1e100, np.inf)])
    points = np.array([[0., 0., 0.], [radii[1], 0., 0.]])
    result = neighborhood_profiles(points, points[[0]], [None], radii)
    assert_array_equal(result['hard_count'], [[1, 2]])
    assert_array_equal(result['count_secant_status'], [['OK']])
    assert_array_equal(result['mass_secant_status'], [['OK']])
    assert np.isfinite(result['count_secants']).all()
    assert np.isfinite(result['mass_secants']).all()
    expected = np.log(2.) / np.log1p((radii[1] - radii[0]) / radii[0])
    assert_allclose(result['count_secants'][0, 0], expected, **TOL)


def test_nonfinite_secant_is_not_reported_ok():
    from manylatents.utils.neighborhood_profiles import _secants
    result, status = _secants(np.array([[1., np.inf]]), RADII[:2], 'ZERO_COUNT')
    assert status[0, 0] == 'NONFINITE_SECANT'
    assert np.isnan(result[0, 0])


def test_self_only_influence_stays_in_physical_patch():
    weights = np.array([[3., 0., 0.]])
    distances = np.array([[0., 1., 2.]])
    physical_patch = np.array([[True, True, False]])  # focal union neighbor 1
    result = influence_profiles(weights, distances, RADII,
                                patch_mask=physical_patch, self_indices=[0])
    assert result['patch_leakage'][0] == 0
    assert result['self_mass'][0] == 3
    assert result['self_fraction'][0] == 1
    nonself_targets = physical_patch.copy()
    nonself_targets[0, 0] = False
    target_result = influence_profiles(weights, distances, RADII, patch_mask=nonself_targets)
    assert target_result['patch_leakage'][0] == 1  # Different target-set estimand.
