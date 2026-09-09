"""Deterministic policy contracts; no experiments or downstream fixture imports."""
import inspect
import numpy as np
import pytest
from manylatents.metrics.task_selection import (CenteredSelectionBall, SelectionView,
    StratifiedSeedCentroid,WeightedCoverageRadius,DistinctGroupBudgetBall,DonorEligibilityByAnswerToken)
from manylatents.utils.task_errors import ContractViolation,NumericalFailure


def f(x):return np.asarray(x,dtype=np.float64)
def ids(x):return np.asarray(x,dtype=np.int64)


def ball(**overrides):
    kw=dict(embeddings=f([[0],[1],[2]]),center=f([0]),ids=ids([20,10,30]),candidate_ids=ids([20,10,30]),threshold_squared=1)
    kw.update(overrides);return CenteredSelectionBall(**kw)


def test_arbitrary_center_unsorted_identity_and_boundary():
    out=ball();assert out['selected_ids'].tolist()==[10,20]
    assert out['selected_mask'].tolist()==[True,True,False]
    assert out['boundary_tie_ids'].tolist()==[10]
    assert ball(exclude_ids=ids([20]))['selected_ids'].tolist()==[10]
    x=f([[0,0],[2,0],[1,1],[1,-1],[3,0],[1,.5],[2,2]])
    out=CenteredSelectionBall(x,center=f([1,0]),ids=ids(range(10,17)),candidate_ids=ids(range(10,17)),threshold_squared=1)
    assert out['selected_ids'].tolist()==[10,11,12,13,15]
    np.testing.assert_array_equal(out['squared_distances'],[1,1,1,1,4,.25,5])


@pytest.mark.parametrize('override',[
    {'ids':ids([20,20,30])},{'candidate_ids':ids([20,20])},{'candidate_ids':ids([99])},
    {'exclude_ids':ids([99])},{'ids':f([20,10,30])},{'center':f([0,1])},
    {'center':np.array([0],dtype=np.float32)},{'embeddings':np.array([[0],[1],[2]],dtype=np.float32)},
    {'threshold_squared':True},{'threshold_squared':-1},{'threshold_squared':float('nan')},
    {'metric_form':np.eye(1)},{'candidate_weights':f([1,-1,1])},{'candidate_weights':f([1,float('inf'),1])},
    {'candidate_groups':ids([0,1])}])
def test_invalid_contract(override):
    with pytest.raises(ContractViolation):ball(**override)


def test_numerical_overflow_typed():
    with pytest.raises(NumericalFailure):ball(embeddings=f([[1e308],[0],[0]]))
    with pytest.raises(NumericalFailure):ball(candidate_weights=f([1e308,1e308,0]))


def test_hash_and_input_mutation():
    x=f([[0],[1],[2]]);i=ids([20,10,30]);before=x.copy()
    a=ball(embeddings=x,ids=i);b=ball(embeddings=x,ids=i)
    assert a['input_hash']==b['input_hash'];assert a['policy_hash']==b['policy_hash']
    assert ball(candidate_weights=f([2,1,1]))['input_hash']!=a['input_hash']
    assert ball(threshold_squared=2)['policy_hash']!=a['policy_hash']
    np.testing.assert_array_equal(before,x)
    a['squared_distances'][0]=42;np.testing.assert_array_equal(before,x)


@pytest.mark.parametrize('field',['outcome','label','fate','effect','logit','counts','target_mask','reference_ids','unreviewed_extra'])
def test_view_unknown_and_forbidden(field):
    with pytest.raises(ContractViolation):SelectionView(ids=ids([1]),coordinates=f([[0]]),candidate_mask=np.array([True]),**{field:'sentinel'})


def test_view_immutable_and_no_target_parameter():
    x=f([[0]]);p={'nested':{'choices':[1,2]}}
    view=SelectionView(ids=ids([1]),coordinates=x,candidate_mask=np.array([True]),policy_params=p)
    x[0]=2;p['nested']['choices'][0]=99
    assert view.coordinates[0,0]==0;assert view.policy_params['nested']['choices']==(1,2)
    with pytest.raises(ValueError):view.coordinates.setflags(write=True)
    with pytest.raises(TypeError):view.policy_params['new']=1
    assert 'target_mask' not in inspect.signature(CenteredSelectionBall).parameters


def test_balanced_centroid_and_coverage():
    x=f([[0],[2],[8]]);i=ids([0,1,2]);kwargs=dict(ids=i,strata=['A','A','B'],required_strata=['A','B'])
    c=StratifiedSeedCentroid(x,**kwargs)
    np.testing.assert_array_equal(c['center'],[4.5]);np.testing.assert_array_equal(c['stratum_centroids'],[[1],[8]])
    np.testing.assert_array_equal(c['seed_weights'],[.25,.25,.5])
    out=WeightedCoverageRadius(x,center=c['center'],rho=.75,**kwargs)
    assert out['threshold_squared']==12.25;assert out['selected_seed_ids'].tolist()==[1,2];assert out['selected_seed_mass']==.75
    np.testing.assert_array_equal(out['squared_distances'],[20.25,6.25,12.25])
    assert WeightedCoverageRadius(x,center=c['center'],rho=1,**kwargs)['selected_seed_ids'].tolist()==[0,1,2]


def test_empty_stratum_and_zero_seed_pool():
    out=StratifiedSeedCentroid(f([[0]]),ids=ids([1]),strata=['A'],required_strata=['A','B'])
    assert out['status']=='ABSTAIN_EMPTY_STRATUM';assert out['center'] is None
    out=StratifiedSeedCentroid(np.empty((0,1),dtype=np.float64),ids=ids([]),strata=np.array([],dtype='U1'),required_strata=['A'])
    assert out['center'] is None


@pytest.mark.parametrize('rho',[0,-1,1.1,True,float('inf')])
def test_rho_rejected(rho):
    with pytest.raises(ContractViolation):WeightedCoverageRadius(f([[0]]),center=f([0]),ids=ids([0]),strata=['A'],required_strata=['A'],rho=rho)


def group_ball(x,g,K=2,candidates=None):
    i=ids(range(len(x)))
    return DistinctGroupBudgetBall(f(x),center=f([0]),ids=i,candidate_ids=i if candidates is None else ids(candidates),group_ids=ids(g),K_groups=K)


def test_group_minimum_not_member_count():
    out=group_ball([[1],[10],[2],[3]],[7,7,8,9])
    assert out['selected_group_ids'].tolist()==[7,8];assert out['selected_ids'].tolist()==[0,2]
    np.testing.assert_array_equal(out['group_min_q'],[1,4,9])
    assert out['threshold_squared']==4


def test_tie_abstention_never_executed_mask():
    out=group_ball([[1],[2],[2]],[0,1,2])
    assert out['status']=='ABSTAIN_TIE_OVERFLOW';assert out['selected_mask'] is None
    assert out['selected_ids'] is None;assert out['threshold_squared'] is None
    assert out['candidate_threshold_squared']==4;assert out['candidate_group_count']==3
    assert out['boundary_tie_ids'].tolist()==[1,2]
    assert out['diagnostic_candidate_mask'].all()


def test_insufficient_and_noneligible_geometric_particles():
    out=group_ball([[1]],[0]);assert out['status']=='ABSTAIN_INSUFFICIENT_GROUPS';assert out['threshold_squared'] is None
    out=group_ball([[1],[2],[0]],[0,1,2],candidates=[0,1]);assert out['selected_ids'].tolist()==[0,1]
    assert out['noneligible_inside_ids'].tolist()==[2]


def test_donor_token_keys_unsorted_ids():
    out=DonorEligibilityByAnswerToken(ids=ids([30,20,10,40]),donor_name_pairs=ids([[11,22],[11,22],[22,11],[11,33]]),
        donor_answer_tokens=ids([11,22,11,11]),recipient_pair=ids([11,22]),recipient_desired_token=11)
    assert out['eligible_ids'].tolist()==[10,30];assert out['rejected_ids'].tolist()==[20,40]
    assert out['rejection_reasons'][20]=='ANSWER_TOKEN_MISMATCH'
    assert out['rejection_reasons'][40]=='NAME_PAIR_MISMATCH'


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_unsigned_group_labels_are_supported(dtype):
    from manylatents.metrics.task_selection import CenteredSelectionBall, SelectionView
    coordinates=np.array([[0.],[1.]],dtype=np.float64)
    ids=np.array([0,1],dtype=np.int64)
    labels=np.array([3,4],dtype=dtype)
    view=SelectionView(ids=ids,coordinates=coordinates,candidate_mask=np.array([True,True]),group_labels=labels)
    assert np.array_equal(view.group_labels,labels)
    record=CenteredSelectionBall(coordinates,center=np.array([0.]),ids=ids,candidate_ids=ids,
        threshold_squared=1.,candidate_groups=labels)
    assert record['group_counts']=={3:1,4:1}


def test_unsigned_group_labels_above_int64_fail_with_contract_error():
    from manylatents.metrics.task_selection import SelectionView
    with pytest.raises(ContractViolation):
        SelectionView(ids=np.array([0],dtype=np.int64),coordinates=np.array([[0.]]),
            candidate_mask=np.array([True]),group_labels=np.array([2**63],dtype=np.uint64))
