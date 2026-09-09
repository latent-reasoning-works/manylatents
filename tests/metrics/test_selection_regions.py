"""Deterministic unit objects; these are not the twelve downstream result scenes."""
import json
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from manylatents.metrics.selection_regions import (
    SelectionRegionProfile, BallMembership, DeclaredEdgeProfile, SelectionGaugeProfile,
)
from manylatents.metrics.registry import get_metric


def base():
    return np.array([[0.,0.],[1.,0.],[2.,0.],[0.,3.]]), np.array([False,True,False,False]), np.eye(2)[None]


def test_exact_thresholds_closed_counts_and_registry():
    Y,S,M = base()
    p = get_metric('selection_region_profile')(Y,target_mask=S,probe_index=0,forms=M)
    assert_array_equal(p['threshold'],[0.,1.,4.,9.])
    assert_array_equal(p['tp'],[0,1,1,1]); assert_array_equal(p['fp'],[0,0,1,2])
    assert_array_equal(p['fn'],[1,0,0,0]); assert_array_equal(p['tn'],[2,2,1,0])
    assert_array_equal(p['tie_count'],[0,1,1,1])
    assert_array_equal(p['boundary_count'],p['tie_count'])
    assert p['best_full_recovery']['candidate_index'] == [1]
    assert p['collision']['floor'] == 0
    assert_array_equal(p['frontier_indices'],[1])
    assert p['contamination'][0] == 0 and not p['contamination_defined'][0]
    assert p['selection_empty'][0] and p['selection_full'][-1]


def test_anchor_duplicate_prevents_fake_empty_and_collision_floor():
    Y = np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])
    S = np.array([True,False,True,False])
    p = SelectionRegionProfile(Y,S,0,np.eye(2)[None])
    assert_array_equal(p['selected_mask'][0],[False,True,False,False])
    assert p['fp'][0] == 1 and not p['selection_empty'].any()
    assert p['collision']['floor'] == 1
    assert (p['fp']+p['fn'] >= p['collision']['floor']).all()
    assert p['collision']['class_id'][0] == -1
    assert p['target_mask'][0] == False
    ball = BallMembership(Y,0,0.)
    assert_array_equal(ball['physical_mask'],[True,True,False,False])
    assert_array_equal(ball['target_mask'],[False,True,False,False])
    assert ball['boundary_count'] == 1


def test_near_ties_are_distinct_and_close_coordinates_not_collisions():
    Y = np.array([[0.,0.],[1.,0.],[np.nextafter(1.,2.),0.]])
    p = SelectionRegionProfile(Y,np.array([False,True,False]),0,np.eye(2)[None])
    assert len(p['threshold']) == 3
    assert p['threshold'][1] < p['threshold'][2]
    assert p['collision']['floor'] == 0
    assert p['best_full_recovery']['fp'][0] == 0


def test_slab_psd_thresholds_include_ineligible_and_boundary_union():
    Y = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,1.],[1.,0.,.25]])
    S = np.array([False,True,False,True])
    M = np.array([np.diag([1.,1.,0.])]*2)
    slab = dict(normal=[0.,0.,1.],offset=0.,halfwidth=.25)
    p = SelectionRegionProfile(Y,S,0,M,slabs=[None,slab],allow_semidefinite=[True,True])
    assert_array_equal(p['threshold'],[0,1,4,0,1,4])
    assert_array_equal(p['fp'],[0,0,1,0,0,0])
    assert_array_equal(p['boundary_count'][3:],[0,2,1])
    assert p['best_full_recovery']['fp'].tolist() == [0,0]
    assert p['tie_count'][-1] == 1
    with pytest.raises(ValueError): SelectionRegionProfile(Y,S,0,M)


def test_unattainable_slab_and_additional_do_not_replace_primary():
    Y,S,M = base()
    slab = dict(normal=[1.,0.],offset=0.,halfwidth=.5)
    p = SelectionRegionProfile(Y,S,0,M,slabs=[slab],thresholds=[[.5,2.,2.]])
    assert_array_equal(p['threshold'],[0,1,4,9])
    assert_array_equal(p['additional']['threshold'],[.5,2.])
    assert not p['additional']['exhaustive'] and p['exhaustive']
    assert p['best_full_recovery']['candidate_index'] == [None]
    assert p['best_full_recovery']['status'] == ['UNATTAINABLE']
    assert p['max_recovery']['candidate_index'] == [0]
    assert p['max_recovery']['recovery'][0] == 0
    assert p['max_recovery']['recovery_defined'][0]
    assert not p['max_recovery']['contamination_defined'][0]


def test_empty_full_and_singleton_universes():
    Y,S,M = base()
    empty = SelectionRegionProfile(Y,np.zeros(4,dtype=bool),0,M,thresholds=[[]])
    assert not empty['recovery_defined'].any()
    assert empty['best_full_recovery']['status'] == ['TARGET_EMPTY']
    assert empty['additional']['selected_mask'].shape == (0,4)
    full = SelectionRegionProfile(Y,np.ones(4,dtype=bool),0,M)
    assert not full['fp'].any() and full['best_full_recovery']['tp'][0] == 3
    single = SelectionRegionProfile(Y[:1],np.array([True]),0,M)
    assert_array_equal(single['threshold'],[0])
    assert single['collision']['floor'] == 0
    assert not single['selected_mask'].any()


def test_ellipse_recovers_stretched_object_and_tied_frontier_retained():
    Y = np.array([[0.,0.],[1.,0.],[0.,.75]])
    S = np.array([False,True,False])
    M = np.array([np.eye(2),np.diag([1.,4.]),np.diag([1.,4.])])
    p = SelectionRegionProfile(Y,S,0,M)
    assert_array_equal(p['best_full_recovery']['fp'],[1,0,0])
    assert len(p['frontier_indices']) == 2
    assert_array_equal(p['form_index'][p['frontier_indices']],[1,2])
    assert len(p['ball_frontier_indices']) == 2  # empty and full-recovery tradeoff


def test_ball_eligibility_and_edge_identity_validation():
    Y,S,M = base()
    b = get_metric('ball_membership')(Y,probe_index=0,radius=2.,eligible_mask=np.array([True,True,False,True]))
    assert_array_equal(b['target_mask'],S)
    e = DeclaredEdgeProfile(Y,np.array([[0,1],[1,2],[0,3]]),1.)
    assert_array_equal(e['retained_mask'],[True,True,False])
    assert e['boundary_count'] == 2 and e['lost_count'] == 1
    for edges in ([[0,0]],[[0,4]],[[0,1],[1,0]],[[0.,1.]]):
        with pytest.raises(ValueError): DeclaredEdgeProfile(Y,edges,1.)
    assert DeclaredEdgeProfile(Y,np.empty((0,2),dtype=int),1.)['lost_count'] == 0


def test_gauge_safe_gaps_and_particle_permutation():
    Y,S,M = base()
    R = np.array([[0.,-1.],[1.,0.]])
    p = SelectionRegionProfile(Y,S,0,M)
    Yr = Y@R.T + [2.,-1.]
    Mr = np.array([R@m@R.T for m in M])
    r = SelectionRegionProfile(Yr,S,0,Mr)
    g = SelectionGaugeProfile(p['q'],r['q'],counted_mask=p['counted_mask'])
    assert g['q_agreement'].all() and g['eligible_count'] == 2
    p2 = SelectionRegionProfile(Y,S,0,M,thresholds=g['gap_thresholds'])
    r2 = SelectionRegionProfile(Yr,S,0,Mr,thresholds=g['gap_thresholds'])
    assert_array_equal(p2['additional']['selected_mask'],r2['additional']['selected_mask'])
    perm = np.array([2,0,3,1])
    pp = SelectionRegionProfile(Y[perm],S[perm],1,M)
    assert_array_equal(pp['selected_mask'],p['selected_mask'][:,perm])
    assert_array_equal(pp['tp'],p['tp'])


@pytest.mark.parametrize('field,value',[
    ('Y',[[0.,0.],[np.nan,1.]]),('Y',[[0j,0j],[1j,0j]]),
    ('M',[[[1.,.1],[0.,1.]]]),('M',[[[1.,0.],[0.,-1.]]]),
    ('M',[[[np.inf,0.],[0.,1.]]]),('M',[[[1j,0],[0,1]]]),
    ('S',[0,1,0,0]),('probe',True),('probe',-1),('probe',1.5),
])
def test_invalid_domains(field,value):
    Y,S,M = base(); args=dict(embeddings=Y,target_mask=S,probe_index=0,forms=M)
    args[dict(Y='embeddings',M='forms',S='target_mask',probe='probe_index')[field]]=value
    with pytest.raises(ValueError): SelectionRegionProfile(**args)


def test_invalid_slabs_thresholds_and_overflow():
    Y,S,M = base()
    for slab in [dict(normal=[0.,0.],offset=0.,halfwidth=1.),dict(normal=[1.,0.],offset=0.,halfwidth=-1.),dict(normal=[1.,0.],halfwidth=1.)]:
        with pytest.raises(ValueError): SelectionRegionProfile(Y,S,0,M,slabs=[slab])
    for t in [[[-1.]],[[np.inf]],[[1j]],[],1.,np.array(1.)]:
        with pytest.raises(ValueError): SelectionRegionProfile(Y,S,0,M,thresholds=t)
    for slabs in (1.,np.array(1.)):
        with pytest.raises(ValueError): SelectionRegionProfile(Y,S,0,M,slabs=slabs)
    with pytest.raises(FloatingPointError): SelectionRegionProfile(Y*1e200,S,0,M)
    with pytest.raises(ValueError): BallMembership(Y,0,-1.)
    with pytest.raises(ValueError): DeclaredEdgeProfile(Y,[[0,1]],np.nan)


def test_new_instrument_output_has_no_nonfinite_placeholders():
    Y,S,M = base()
    p = SelectionRegionProfile(Y,np.zeros(4,dtype=bool),0,M,thresholds=[[]])
    def safe(x):
        if isinstance(x,np.ndarray): return safe(x.tolist())
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [safe(v) for v in x]
        return x
    json.dumps(safe(p),allow_nan=False)


def test_declared_seventeen_forms_and_transport_of_slab_units():
    # Generic four-particle unit object, not a downstream scene evaluation.
    Y,S,_ = base()
    forms = [np.eye(2)]
    for axis_ratio in (2.,4.):
        for j in range(8):
            theta = j*np.pi/8
            R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            M = R @ np.diag([1.,axis_ratio**2]) @ R.T
            forms.append((M+M.T)/2)
    p = SelectionRegionProfile(Y,S,0,forms)
    assert p['q'].shape == (17,4)
    assert len(p['offsets']) == 18
    for f in range(17):
        rows = slice(p['offsets'][f],p['offsets'][f+1])
        assert_array_equal(p['threshold'][rows],np.unique(np.r_[0,p['q'][f,1:]]))
    normal = np.array([1.,0.]); shift = np.array([2.,-1.])
    R = np.array([[0.,-1.],[1.,0.]])
    slab = dict(normal=normal,offset=0.,halfwidth=1.5)
    transformed_normal = R@normal
    moved_slab = dict(normal=transformed_normal,offset=float(transformed_normal@shift),halfwidth=1.5)
    original = SelectionRegionProfile(Y,S,0,[np.eye(2)],slabs=[slab])
    moved = SelectionRegionProfile(Y@R.T+shift,S,0,[np.eye(2)],slabs=[moved_slab])
    assert_array_equal(original['selected_mask'],moved['selected_mask'])
    assert_array_equal(original['boundary_count'],moved['boundary_count'])


def test_gauge_excludes_focal_and_unsafe_gaps():
    reference = np.array([[0.,1.,1.+1e-12,3.]])
    moved = reference.copy(); moved[0,-1] += 1e-11
    g = SelectionGaugeProfile(reference,moved,counted_mask=np.array([False,True,True,True]))
    assert g['eligible_count'] == 1
    assert len(g['gap_thresholds'][0]) == 1
    assert 1. < g['gap_thresholds'][0][0] < 3.
    for bad in (np.array([[np.nan,1,2,3]]), np.array([[0j,1j,2j,3j]])):
        with pytest.raises(ValueError): SelectionGaugeProfile(reference,bad,counted_mask=np.ones(4,dtype=bool))
