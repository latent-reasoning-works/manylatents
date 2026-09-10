import numpy as np
import pytest
from manylatents.metrics.task_summaries import SignedWeightedAttributeAccount as account, weighted_total_variation
from manylatents.utils.task_errors import ContractViolation, NumericalFailure


def fixture(ref=(True, True, False), sel=(True, False, False), weights=(1., 1., 1.), field=(0., 1., .5)):
    return dict(ids=np.array([10,20,30],dtype=np.int64), reference_mask=np.array(ref),
                selected_mask=np.array(sel), weights=np.array(weights), attribute=np.array(field), attribute_range=np.array([0.,1.]))


def check_integrity(k):
    snapshots={n:v.tobytes() for n,v in k.items()}
    a=account(**k)
    assert all(v.tobytes()==snapshots[n] for n,v in k.items())
    if a['status']=='OK':
        for key in ['membership_reconstruction_residual','residual_reconstruction_residual','direct_centered_residual']:
            assert abs(a[key])<=1e-12
        assert a['absolute_error']<=a['bound']+1e-12
        old=weighted_total_variation(ids=a['ids'],mass_p=a['raw_mass_p'],mass_q=a['raw_mass_q'],attribute=k['attribute'],attribute_range=k['attribute_range'])
        assert old['TV']==pytest.approx(a['TV'],abs=1e-12)
        assert old['observed_weighted_attribute_error']==pytest.approx(a['absolute_error'],abs=1e-12)
    return a


def test_omit_and_permutation():
    k=fixture();a=check_integrity(k)
    for key,value in dict(mean_reference=.5,mean_selected=0,signed_error=-.5,TV=.5,residual_contrast=-1,membership_reconstruction=-.5).items():
        assert a[key]==pytest.approx(value,abs=1e-12)
    assert a['removed_ids'].tolist()==[20] and a['added_ids'].tolist()==[]
    assert a['removed_contributions'].tolist()==[0.,-.5,0.]
    b=check_integrity({n:v[::-1] if n!='attribute_range' else v for n,v in k.items()})
    for name,value in a.items():
        if isinstance(value,float):assert value==pytest.approx(b[name],abs=1e-12)
    for name in ['normalized_p','normalized_q','excess','deficit','added_contributions','removed_contributions']:
        np.testing.assert_array_equal(a[name],b[name][::-1])


@pytest.mark.parametrize('constant',[False,True])
def test_cancel(constant):
    a=check_integrity(fixture(ref=(False,True,False),sel=(True,True,True),field=(.5,.5,.5) if constant else (0.,.5,1.)))
    assert a['mean_reference']==a['mean_selected']==.5
    assert abs(a['signed_error'])<=1e-12 and a['TV']==pytest.approx(2/3)
    assert a['residual_contrast']==pytest.approx(0,abs=1e-12)
    if not constant:np.testing.assert_allclose(a['added_contributions'],[-1/6,0,1/6],atol=1e-12)


def test_zero_tv_and_zero_weight_addition():
    a=check_integrity(fixture(sel=(True,True,False),weights=(1.,2.,1.)))
    assert a['selected_mass']==a['reference_mass']==3
    assert a['mean_selected']==pytest.approx(2/3)
    assert a['TV']==0 and a['residual_status']=='NOT_APPLICABLE_ZERO_TV'
    assert a['excess'] is a['deficit'] is a['residual_contrast'] is None
    b=check_integrity(fixture(ref=(True,False,False),sel=(True,True,False),weights=(1.,0.,1.)))
    assert b['TV']==0 and b['added_ids'].tolist()==[20]


@pytest.mark.parametrize('ref,sel,w',[((False,False,False),(True,False,False),(1.,1.,1.)),((True,False,False),(False,False,False),(1.,1.,1.)),((True,False,False),(True,True,False),(0.,0.,0.))])
def test_zero_mass(ref,sel,w):
    a=check_integrity(fixture(ref=ref,sel=sel,weights=w))
    assert a['status']=='NOT_EVALUABLE_ZERO_MASS'
    for key in ['mean_reference','mean_selected','signed_error','TV','bound','normalized_p','normalized_q','membership_reconstruction','residual_contrast']:
        assert a[key] is None


def test_historical_f14():
    a=check_integrity(fixture(ref=(True,True,False),sel=(True,True,True),weights=(1.,1.,98.),field=(0.,0.,1.)))
    assert a['reference_mass']==2 and a['selected_mass']==100
    assert a['signed_error']==a['TV']==.98


@pytest.mark.parametrize('key,value',[
    ('ids',np.array([10,10,30],dtype=np.int64)),('ids',np.array([10,20,30],dtype=np.uint64)),
    ('reference_mask',np.array([1,1,0],dtype=np.int64)),('selected_mask',np.array([True,False])),
    ('weights',np.array([1.,-1.,1.])),('weights',np.array([1.,np.nan,1.])),
    ('attribute',np.array([0.,np.nan,.5])),('attribute',np.array([0.,2.,.5])),('attribute_range',np.array([1.,0.])),
    ('ids',np.array([True,False,True])),('attribute_range',np.array([-np.inf,1.])),
])
def test_contract_guards(key,value):
    k=fixture();k[key]=value;snapshots={n:v.tobytes() for n,v in k.items()}
    with pytest.raises(ContractViolation):account(**k)
    assert all(v.tobytes()==snapshots[n] for n,v in k.items())


def test_finite_overflows_and_copies():
    k=fixture(ref=(True,True,False),sel=(True,True,False),weights=(np.finfo(float).max,np.finfo(float).max,1.))
    with pytest.raises(NumericalFailure):account(**k)
    a=check_integrity(fixture());a['weights'][0]=99
    assert fixture()['weights'][0]==1
    k=fixture(field=(-np.finfo(float).max,0.,np.finfo(float).max));k['attribute_range']=np.array([-np.finfo(float).max,np.finfo(float).max])
    with pytest.raises(NumericalFailure):account(**k)


def test_equiv_and_registration():
    from manylatents.metrics.task_selection import CenteredSelectionBall
    from manylatents.metrics.selection_regions import SelectionRegionProfile
    from manylatents.metrics.registry import get_metric
    assert get_metric('signed_weighted_attribute_account').func is account
    x=np.array([[.125,0],[0,.125],[0,.5],[.5,0],[-.125,.125],[0,0]],dtype=np.float64)
    y=x*[1.,.25];z=y*[1.,4.]
    kwargs=dict(center=np.zeros(2),ids=np.arange(6,dtype=np.int64),candidate_ids=np.arange(5,dtype=np.int64),exclude_ids=np.array([5],dtype=np.int64),threshold_squared=.0625)
    ref=CenteredSelectionBall(x,**kwargs);repair=CenteredSelectionBall(z,**kwargs)
    curve=SelectionRegionProfile(y,ref['selected_mask'],5,np.array([np.diag([1.,16.])]),thresholds=[np.array([.0625])])
    assert repair['selected_ids'].tolist()==[0,1,4]
    np.testing.assert_array_equal(curve['additional']['selected_mask'][0],repair['selected_mask'])
