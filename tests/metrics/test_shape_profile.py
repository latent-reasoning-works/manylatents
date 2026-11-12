"""Shape profiles use exact parent geometry and fixed rows."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from manylatents.metrics.shape_profile import shape_profile, _exact_knn, _window


def cloud(n=600):
    return np.random.default_rng(12).normal(size=(n, 5))


def row(rows, name, k=None, aggregation='value', kind='dataset'):
    return next(r for r in rows if (r['entity_kind'], r['stat_name'], r['scale_value'], r['aggregation']) == (kind, name, k, aggregation))


def test_600_points_have_fixed_rows_and_censor_large_k():
    rows = shape_profile(cloud())
    assert len(rows) == 15
    for r in rows:
        if r['scale_value'] in (80, 160):
            assert r['status'] == 'censored' and r['value'] is None
        assert r['value'] is None or type(r['value']) is float and np.isfinite(r['value'])
    assert row(rows, 'n_cells')['value'] == 600
    assert rows == shape_profile(cloud())


def test_selection_aggregates_parent_values():
    x = cloud()
    selected = np.arange(len(x)) < 30
    rows = shape_profile(x, selection=selected)
    assert len(rows) == 56
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    expected = np.median(np.log(d[selected, 4] / d[selected, 14]))
    assert row(rows, 'log_rk_r15', 5, 'median', 'selection')['value'] == pytest.approx(expected, abs=1e-12)
    assert row(rows, 'selection_fidelity', 5, kind='selection')['reason'] == 'layout_unavailable'
    delta = row(rows, 'relative_log_rk_r15', 5, 'median', 'selection')
    assert delta['value'] == pytest.approx(expected - row(rows, 'log_rk_r15', 5, 'median')['value'], abs=1e-12)


def test_exact_float64_log_ratio_matches_direct_oracle():
    x = cloud(2000)
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    expected = np.median(np.log(d[:, 4] / d[:, 14]))
    assert row(shape_profile(x), 'log_rk_r15', 5, 'median')['value'] == pytest.approx(expected, abs=1e-12)
    assert row(shape_profile(x * 1e-100), 'log_rk_r15', 5, 'median')['value'] == pytest.approx(expected, abs=1e-12)


def test_ties_exclude_self_and_choose_lower_index():
    _, indices = _exact_knn(np.zeros((5, 2)), 3)
    assert indices.tolist() == [[1,2,3], [0,2,3], [0,1,3], [0,1,2], [0,1,2]]


def test_windows_longest_tie_gap_and_boundary():
    assert _window({5: 2., 10: 2.1, 20: 8., 40: 8.1}, (5,10,20,40)) == (5,10)
    assert _window({5: 2., 20: 2.}, (5,20)) is None
    rows = shape_profile(cloud())
    hi = row(rows, 'window_hi')
    if hi['value'] == 40:
        assert hi['status'] == 'censored'
        assert row(rows, 'window_lo')['status'] == 'ok'


@pytest.mark.parametrize('nselected', [0, 19, 20])
def test_duplicates_and_selection_count(nselected):
    x = cloud()
    x = np.concatenate([x, x[:30]])
    selected = np.zeros(len(x), dtype=bool)
    selected[:nselected] = True
    selected[600:600+nselected] = True
    rows = shape_profile(x, selection=selected)
    assert row(rows, 'n_cells')['value'] == 630
    count = row(rows, 'n_cells', kind='selection')
    assert count['value'] == nselected * 2 and count['status'] == 'ok'
    assert count['n_cells_used'] == nselected
    assert row(rows, 'log_rk_r15', 5, 'median', 'selection')['status'] == ('undefined' if nselected < 20 else 'ok')


@pytest.mark.parametrize('x', [np.empty((0,2)), np.empty((5,0)), np.zeros((50,2)), np.full((50,2), np.nan), np.full((50,2), np.inf)])
def test_unresolved_fixed_rows(x):
    rows = shape_profile(x)
    assert len(rows) == 15
    assert row(rows, 'n_cells')['status'] == 'ok'
    assert all(r['value'] is None or np.isfinite(r['value']) for r in rows)


def test_extended_refusal():
    with pytest.raises(ValueError, match='not_implemented'):
        shape_profile(cloud(), extended=True)


def test_conductance_and_fidelity_match_direct_edges():
    x = cloud()
    layout = x[:, :2]
    selected = np.arange(600) % 3 == 0
    rows = shape_profile(x, selection=selected, layout=layout)
    def retained(points):
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        edges = np.argsort(d, axis=1, kind='stable')[:, :5]
        return selected[edges[selected]].mean()
    high, low = retained(x), retained(layout)
    assert row(rows, 'conductance', 5, 'share', 'selection')['value'] == 1-high
    assert row(rows, 'selection_fidelity', 5, kind='selection')['value'] == high/low


@pytest.mark.parametrize('n,status', [(49,'censored'), (50,'ok'), (99,'censored'), (100,'ok')])
def test_scale_threshold(n, status):
    k = 5 if n < 60 else 10
    assert row(shape_profile(cloud(n)), 'log_rk_r15', k, 'median')['status'] == status


def test_window_endpoint_retains_lower_bound():
    from manylatents.metrics.shape_profile import _core_rows
    logs = {k:np.ones(600) for k in (5,10,20,40)}
    lids = {k:np.ones(600)*2 for k in (5,10,20,40)}
    rows = _core_rows(logs,lids,np.arange(600),kind='dataset',n_cells=600,n_distinct=600,failure='')
    lo, hi = row(rows,'window_lo'), row(rows,'window_hi')
    assert (lo['status'],lo['value']) == ('ok',5.)
    assert (hi['status'],hi['value'],hi['reason']) == ('censored',40.,'window_right_censored')


def test_relative_prefers_undefined_selection_over_censored_parent():
    selected = np.arange(600) < 19
    rows = shape_profile(cloud(), selection=selected)
    r = row(rows,'relative_log_rk_r15',160,'median','selection')
    assert (r['status'],r['reason'],r['value']) == ('undefined','relative_selection_too_few_selected',None)
