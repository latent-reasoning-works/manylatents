"""Finite deterministic objects for quantiles, correspondence and killed mass."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from manylatents.utils.operator_profiles import (
    weighted_radius_quantiles, finite_patch_propagation, transported_patch_summary,
)
from manylatents.utils.neighborhood_profiles import influence_profiles


def test_self_zero_rows_closed_crossing_and_offgrid_quantiles():
    W = np.array([[3.,0,0], [0,0,0], [0,2,2]])
    D = np.array([[0.,1,2], [1,0,2], [3,1.25,0]])
    result = weighted_radius_quantiles(W,D)
    assert_allclose(result['radii'][[0,2]], [[0,0],[0,1.25]])
    assert result['status'][1] == 'ZERO_ROW'
    assert np.isnan(result['radii'][1]).all()
    mask = np.array([[True,False,False],[False,True,False],[False,False,True]])
    influence = influence_profiles(W,D,np.array([.5,2.]),patch_mask=mask,self_indices=[0,1,2])
    assert influence['patch_leakage'][0] == 0
    full = weighted_radius_quantiles(np.array([[1.,1.,0.]]),np.array([[1.,2.,3.]]),quantiles=(1.,))
    assert full['radii'][0,0] == 2


def test_positive_row_rescaling_and_particle_permutation():
    W = np.array([[2.,1,1],[1,2,1],[1,1,2]])
    D = np.array([[0.,1,3],[1,0,2],[3,2,0]])
    base = weighted_radius_quantiles(W,D)
    scaled = weighted_radius_quantiles(W * np.array([1e-200,3,1e200])[:,None],D)
    assert_allclose(base['radii'],scaled['radii'],atol=1e-10,rtol=1e-9)
    perm = [2,0,1]
    shuffled = weighted_radius_quantiles(W[np.ix_(perm,perm)],D[np.ix_(perm,perm)])
    assert_allclose(shuffled['radii'],base['radii'][perm],atol=1e-10,rtol=1e-9)


def test_equal_participation_can_have_different_reach():
    W = np.array([[.5,.5,0,0],[.5,0,0,.5]])
    D = np.array([[0.,1,2,10],[0,1,2,10]])
    profiles = influence_profiles(W,D,np.array([.5,1.,10.]))
    assert_allclose(profiles['participation'],[2,2])
    q = weighted_radius_quantiles(W,D)
    assert_allclose(q['radii'][:,1],[1,10])


def test_three_state_return_chain_against_hand_recursion():
    P = np.array([[.5,.5,0],[.25,.5,.25],[.5,0,.5]])
    result = finite_patch_propagation(P,[0],[[1]],steps=(1,2,3))
    # J={0,1}; outside2 can return to0. Manually propagated killed vectors:
    # [1,0] -> [.5,.5] -> [.375,.5] -> [.3125,.4375].
    assert_allclose(result['killed_loss'],[[0,.125,.25]])
    assert_allclose(result['delta'],[.25])
    assert_allclose(result['endpoint_outside'],[[0,.125,.1875]])
    assert result['endpoint_outside'][0,2] < result['killed_loss'][0,2]
    assert_allclose(result['bound_geom'],[[.25,.4375,.578125]])
    assert_allclose(result['bound_lin'],[[.25,.5,.75]])
    assert result['checks'].all()
    assert result['monotonicity'].all()
    assert_array_equal(result['physical_patch_ids'][0],[0,1])
    assert_allclose(result['raw_powers']['2'],P@P)
    assert_allclose(result['powers']['3'],P@P@P)


def test_full_physical_patch_and_self_only_patch():
    P = np.array([[.5,.5],[.25,.75]])
    result = finite_patch_propagation(P,[0,0],[[1],[]])
    assert_allclose(result['delta'],[0,.5])
    assert_allclose(result['killed_loss'][0],0,atol=1e-10)
    assert_allclose(result['bound_geom'][1],1-.5**np.array([1,5,15]))
    assert result['bound_status'][1,-1] == 'NEAR_VACUOUS'
    assert result['checks'].all()


def test_domain_corrections_reported_and_invalid_domains_unavailable():
    P = np.array([[.5,.5],[.25,.75]])
    perturbed = P.copy(); perturbed[0,0] += 1e-7
    result = finite_patch_propagation(perturbed,[0],[[1]])
    assert result['corrections']['applied']
    assert 0 < result['corrections']['max_entry_correction'] < 1e-6
    assert_allclose(result['raw_powers']['1'],perturbed)
    assert_allclose(result['transition_used'].sum(axis=1),1)
    invalid = finite_patch_propagation(np.array([[0.,0],[.25,.75]]),[0],[[1]])
    assert invalid['status'][0] == 'INVALID_TRANSITION_DOMAIN'
    assert np.isnan(invalid['killed_loss']).all()
    assert not invalid['checks'].any()
    negative = finite_patch_propagation(np.array([[1.1,-.1],[.25,.75]]),[0],[[1]])
    assert negative['transition_used'] is None


def test_transport_intruders_boundary_empty_and_full():
    Y = np.array([[0.,0],[0,0],[1,0],[-1,0],[0,2]])
    result = transported_patch_summary(Y,[0,0,0,0],[[1,2],[4],[],[1,2,3,4]])
    assert_array_equal(result['status'],['BOUNDARY_TIE','INTERLEAVED','EMPTY_PATCH','EMPTY_COMPLEMENT'])
    assert_allclose(result['intruder_count'][[0,1,3]],[1,3,0])
    assert np.isnan(result['intruder_count'][2])
    assert_array_equal(result['strictly_selectable'],[False,False,None,True])


@pytest.mark.parametrize('weights', [np.array([[1.,-1.]]),np.array([[np.inf,1.]]),np.array([[np.nan,1.]])])
def test_signed_or_nonfinite_quantile_weights_rejected(weights):
    with pytest.raises(ValueError):
        weighted_radius_quantiles(weights,np.array([[0.,1.]]))


@pytest.mark.parametrize('kwargs',[{'steps':(0,)},{'steps':(2,1)}, {'steps':(1,1)}, {'steps':(True,)}, {'atol':-1}])
def test_invalid_propagation_arguments(kwargs):
    with pytest.raises(ValueError):
        finite_patch_propagation(np.eye(2),[0],[[1]],**kwargs)
