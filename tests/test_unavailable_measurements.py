"""Unavailable evidence must never turn into a numeric measurement or action."""
from types import SimpleNamespace

import numpy as np
import pytest

from manylatents.algorithms.latent.selective_correction import _compute_mismatch_labels
from manylatents.algorithms.latent.trajectory_aligner import TrajectoryAligner
from manylatents.metrics.alignment_score import stratify_by_percentile
from manylatents.metrics.cka import cka_pairwise
from manylatents.metrics.mismatch_ratio import MismatchRatio
from manylatents.metrics.signal_geometry import layer_geometry
from manylatents.metrics.trajectory_geometry import (
    TrajectoryCurvature, TrajectoryVelocity, compute_menger_curvature,
)
from manylatents.metrics.trustworthiness import Trustworthiness
from manylatents.utils.knn import compute_knn
from manylatents.utils.stats import bootstrap_ci
from manylatents.utils.exceptions import MeasurementUnavailable


@pytest.mark.parametrize('k', [25, 19, 0, -1, 38])
def test_trustworthiness_refuses_invalid_neighborhood(k):
    rng = np.random.default_rng(0)
    high, low = rng.normal(size=(38, 8)), rng.normal(size=(38, 2))
    with pytest.raises(MeasurementUnavailable, match='Trustworthiness'):
        Trustworthiness(low, dataset=SimpleNamespace(data=high), n_neighbors=k)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('include_self', [False, True])
def test_identical_points_exclude_query_by_index(cached, include_self):
    data = np.ones((5, 3))
    cache = {} if cached else None
    if cached:
        compute_knn(data, k=3, cache=cache)
    distances, indices = compute_knn(data, k=2, include_self=include_self, cache=cache)
    if include_self:
        np.testing.assert_array_equal(indices[:, 0], np.arange(5))
        indices = indices[:, 1:]
    assert not np.any(indices == np.arange(5)[:, None])
    assert indices.shape == (5, 2)
    np.testing.assert_array_equal(distances, 0)


class MissingAffinity:
    neighborhood_size = 15

    def affinity(self, **kwargs):
        raise NotImplementedError('no graph')


@pytest.mark.parametrize('module', [
    None, MissingAffinity(), SimpleNamespace(),
    SimpleNamespace(affinity=lambda **kw: None),
    SimpleNamespace(affinity=lambda **kw: np.ones((7, 7))),
    SimpleNamespace(affinity=lambda **kw: np.ones((40, 7))),
])
@pytest.mark.parametrize('correction', [False, True])
def test_missing_or_misaligned_graph_cannot_report_or_target(module, correction):
    data = np.random.default_rng(0).normal(size=(40, 4))
    with pytest.raises(MeasurementUnavailable, match='affinity'):
        if correction:
            _compute_mismatch_labels(data, module, k_max=20)
        else:
            MismatchRatio(data[:, :2], dataset=SimpleNamespace(data=data), module=module, k=20)


def test_default_signal_readout_is_held_out_on_high_dimensional_noise():
    data = np.random.default_rng(0).normal(size=(40, 2000))
    labels = np.repeat([0, 1], 20)
    result = layer_geometry(data, labels)
    held_out = layer_geometry(data, labels, cv=5)
    assert result.auroc == held_out.auroc
    assert result.auroc < 0.8
    assert result.evaluation_mode == 'stratified_cv'
    assert result.cv_folds == 5


def test_in_cohort_signal_readout_is_explicit():
    data = np.random.default_rng(0).normal(size=(40, 2000))
    result = layer_geometry(data, np.repeat([0, 1], 20), cv=None)
    assert result.evaluation_mode == 'in_cohort'
    assert result.cv_folds is None


@pytest.mark.parametrize('failure', ['exception', 'nan', 'inf'])
def test_bootstrap_refuses_17_successes_out_of_10000(failure):
    calls = 0

    def statistic(data):
        nonlocal calls
        calls += 1
        if calls <= 17:
            return float(np.mean(data))
        if failure == 'exception':
            raise ValueError('estimator failed')
        return float(failure)

    with pytest.raises(MeasurementUnavailable, match='17/10000'):
        bootstrap_ci(statistic, np.arange(20.), n_bootstrap=10000)
    assert calls == 10000


@pytest.mark.parametrize('data', [np.ones((5, 2)), np.zeros((5, 2)),
                                   np.array([[0., 0.], [1., 0.], [0., 0.]])])
def test_degenerate_menger_curvature_is_unavailable(data):
    with pytest.raises(MeasurementUnavailable, match='curvature'):
        compute_menger_curvature(data)


@pytest.mark.parametrize('metric,length', [(TrajectoryCurvature, 2), (TrajectoryVelocity, 1)])
@pytest.mark.parametrize('grouped', [False, True])
def test_short_traces_are_unavailable(metric, length, grouped):
    data = np.ones((length, 2))
    dataset = SimpleNamespace(step_trace_ids=np.zeros(length)) if grouped else None
    with pytest.raises(MeasurementUnavailable, match='trajectory'):
        metric(data, dataset=dataset)


def test_equal_alignment_thresholds_cannot_define_strata():
    with pytest.raises(MeasurementUnavailable, match='threshold'):
        stratify_by_percentile(np.zeros(20))


@pytest.mark.parametrize('reference', [np.zeros((3, 2)), np.ones((3, 2))])
def test_zero_norm_alignment_is_unavailable(reference):
    with pytest.raises(MeasurementUnavailable, match='norm'):
        TrajectoryAligner().residual(np.zeros((3, 2)), reference)


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
@pytest.mark.parametrize('value', [0., 1.])
def test_collapsed_cka_is_unavailable(kernel, value):
    with pytest.raises(MeasurementUnavailable, match='CKA'):
        cka_pairwise(np.full((20, 3), value), np.random.default_rng(0).normal(size=(20, 4)), kernel=kernel)


@pytest.mark.parametrize('k', [None, 0, -1, 5, 6, 1.5, True])
def test_knn_refuses_substituted_neighborhood_even_with_cache(k):
    data = np.ones((5, 2))
    cache = {}
    compute_knn(data, k=4, cache=cache)
    with pytest.raises(MeasurementUnavailable, match='kNN'):
        compute_knn(data, k=k, cache=cache)


def test_valid_trustworthiness_measures_unrelated_data():
    from sklearn.manifold import trustworthiness

    rng = np.random.default_rng(0)
    high, low = rng.normal(size=(100, 8)), rng.normal(size=(100, 2))
    actual = Trustworthiness(low, dataset=SimpleNamespace(data=high), n_neighbors=10)
    assert actual == pytest.approx(trustworthiness(high, low, n_neighbors=10))
    assert actual < 0.8
    assert Trustworthiness(high, dataset=SimpleNamespace(data=high), n_neighbors=10) == 1.0


@pytest.mark.parametrize('bad_weights', [np.zeros((40, 40)), np.full((40, 40), np.nan),
                                        np.full((40, 40), -1.)])
def test_unusable_affinity_weights_are_unavailable(bad_weights):
    data = np.random.default_rng(0).normal(size=(40, 4))
    module = SimpleNamespace(affinity=lambda **kw: bad_weights)
    with pytest.raises(MeasurementUnavailable, match='affinity'):
        MismatchRatio(data[:, :2], dataset=SimpleNamespace(data=data), module=module, k=20)


def test_selective_correction_aborts_before_moving_points(monkeypatch):
    from manylatents.algorithms.latent.selective_correction import SelectiveCorrectionModule
    from manylatents.algorithms.latent.pca import PCAModule

    data = np.random.default_rng(0).normal(size=(40, 4))
    inner = PCAModule(n_components=2)
    monkeypatch.setattr(inner, 'affinity', MissingAffinity().affinity)
    correction = SelectiveCorrectionModule(inner=inner, diagnostic_k=20)

    def forbidden(*args):
        pytest.fail('correction attempted without graph evidence')

    monkeypatch.setattr(correction, '_correct', forbidden)
    with pytest.raises(MeasurementUnavailable, match='affinity'):
        correction.fit_transform(data)
    assert correction._mismatched is None
    assert correction._mismatch_ratio is None
    assert correction.extra_outputs() == {}


@pytest.mark.parametrize('dispatch', ['registry', 'hydra'])
def test_evaluation_propagates_unavailable_without_a_score(dispatch):
    from manylatents.evaluate import evaluate
    from omegaconf import OmegaConf

    data = np.random.default_rng(0).normal(size=(38, 4))
    metrics = ['trustworthiness'] if dispatch == 'registry' else {
        'trust': OmegaConf.create({
            '_target_': 'manylatents.metrics.trustworthiness.Trustworthiness',
            '_partial_': True, 'n_neighbors': 25, 'at': 'embedding',
        })
    }
    with pytest.raises(MeasurementUnavailable, match='Trustworthiness'):
        evaluate(data[:, :2], dataset=SimpleNamespace(data=data), metrics=metrics)


def test_bootstrap_refuses_even_one_failed_requested_replicate():
    calls = 0

    def statistic(data):
        nonlocal calls
        calls += 1
        if calls == 100:
            raise ValueError('one failure')
        return np.mean(data)

    with pytest.raises(MeasurementUnavailable, match='99/100') as failure:
        bootstrap_ci(statistic, np.arange(20.), n_bootstrap=100)
    assert str(failure.value.__cause__) == 'one failure'


@pytest.mark.parametrize('metric', [TrajectoryCurvature, TrajectoryVelocity])
def test_unavailable_trace_cannot_be_silently_dropped(metric):
    data = np.random.default_rng(0).normal(size=(6, 4))
    dataset = SimpleNamespace(step_trace_ids=np.array([0, 0, 0, 0, 0, 1]))
    with pytest.raises(MeasurementUnavailable, match='trajectory'):
        metric(data, dataset=dataset)


def test_cosine_velocity_requires_nonzero_vectors():
    with pytest.raises(MeasurementUnavailable, match='norm'):
        TrajectoryVelocity(np.zeros((5, 2)))


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_valid_cka_retains_identity(kernel):
    data = np.random.default_rng(0).normal(size=(20, 3))
    assert cka_pairwise(data, data, kernel=kernel) == pytest.approx(1.)


def test_non_degenerate_strata_remain_disjoint():
    result = stratify_by_percentile(np.arange(20.))
    assert not np.any(result.aligned_mask & result.divergent_mask)
    assert result.counts == {'divergent': 5, 'middle': 10, 'aligned': 5}


def test_small_cohort_can_override_speculative_prewarm_neighborhood():
    from manylatents.evaluate import evaluate

    data = np.random.default_rng(0).normal(size=(10, 4))
    scores = evaluate(data, dataset=SimpleNamespace(data=data),
                      metrics=['trustworthiness'], n_neighbors=3)
    assert scores['trustworthiness'] == 1.0


@pytest.mark.parametrize('cv', [None, 5])
def test_signal_axis_cannot_be_invented_for_collapsed_vectors(cv):
    data = np.ones((40, 4))
    with pytest.raises(MeasurementUnavailable, match='axis'):
        layer_geometry(data, np.repeat([0, 1], 20), cv=cv)
