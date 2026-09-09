"""Deterministic native-summary controls using literal count/probability arrays."""
import math
import numpy as np
import pytest
from scipy import sparse
from manylatents.metrics.task_summaries import (
    SelectedPseudobulkLogCPM, PairedEffectError, CloneVoteFateDistribution,
    SquaredProbabilityError, weighted_total_variation,
)
from manylatents.utils.task_errors import ContractViolation, NumericalFailure


def bulk(counts, selected=None, **overrides):
    counts=sparse.csr_matrix(counts)
    kwargs=dict(cell_ids=np.arange(counts.shape[1]),gene_ids=np.arange(counts.shape[0]),
                selected_ids=np.arange(counts.shape[1]) if selected is None else selected,
                G_lib=np.arange(counts.shape[0]),G_eval=np.arange(counts.shape[0]))
    kwargs.update(overrides)
    return SelectedPseudobulkLogCPM(counts,**kwargs)


def test_sparse_counts_exact_library_not_all_genes(monkeypatch):
    counts=sparse.csr_matrix([[1,0,1,0,0],[0,1,1,0,0],[0,0,0,2,0],[9,9,9,9,0]])
    def reject(*args,**kwargs):raise AssertionError('Densification forbidden')
    monkeypatch.setattr(sparse.csr_matrix,'toarray',reject)
    r=SelectedPseudobulkLogCPM(counts,cell_ids=[0,1,2,3,4],gene_ids=[0,1,2,3],
                              selected_ids=[0,1,2],G_lib=[0,1,2],G_eval=[0,1,2])
    np.testing.assert_array_equal(r['gene_sums'],[2,2,0,27])
    assert r['library_mass']==4
    np.testing.assert_allclose(r['h'],[math.log2(500001),math.log2(500001),0],atol=1e-12,rtol=0)


def test_high_umi_outsider():
    x=[[0,0,98],[1,1,0]]
    a=bulk(x,G_eval=[0]);b=bulk(x,selected=[0,1],G_eval=[0])
    assert a['library_mass']==100 and b['library_mass']==2
    assert a['h'][0]==math.log2(980001) and b['h'][0]==0
    r=weighted_total_variation(ids=[0,1,2],mass_p=[1,1,0],mass_q=[1,1,98],
                                attribute_range=[0,1],attribute=[0,0,1])
    assert r['TV']==pytest.approx(.98) and r['bounded_attribute_product']==pytest.approx(.98)
    assert r['observed_weighted_attribute_error']==pytest.approx(.98)


def test_integral_float_and_unsorted_identity_maps():
    r=bulk([[2.,3.,7.]],selected=[10,20],cell_ids=[20,10,30],gene_ids=[9],G_lib=[9],G_eval=[9])
    assert r['gene_sums'][0]==5 and r['eval_gene_ids'][0]==9
    np.testing.assert_array_equal(r['selected_ids'],[10,20])


@pytest.mark.parametrize('bad',[.5,-1,float('nan'),float('inf'),1j,True])
def test_bad_count_domain(bad):
    with pytest.raises(ContractViolation):bulk([[bad]])


def test_integer_precision_and_both_accumulation_overflows():
    value=2**53+1
    assert bulk(np.array([[value]],dtype=np.int64))['gene_sums'][0]==value
    limit=np.iinfo(np.int64).max
    with pytest.raises(NumericalFailure):bulk(np.array([[limit,1]],dtype=np.int64))
    with pytest.raises(NumericalFailure):bulk(np.array([[limit],[1]],dtype=np.int64))
    with pytest.raises(NumericalFailure):bulk([[float(2**63)]])


def test_sparse_duplicate_entries_do_not_wrap_before_validation():
    x=sparse.coo_matrix((np.array([2**63-1,1],dtype=np.int64),([0,0],[0,0])),shape=(1,1))
    with pytest.raises(NumericalFailure):SelectedPseudobulkLogCPM(x,cell_ids=[0],gene_ids=[0],selected_ids=[0],G_lib=[0],G_eval=[0])


def test_statuses_and_no_invalid_zero_observation():
    r=bulk([[0]],min_cells=2)
    assert r['status']=='NOT_EVALUABLE_MIN_CELLS' and r['h'] is None
    r=bulk([[0]])
    assert r['status']=='NOT_EVALUABLE_ZERO_LIBRARY' and r['h'] is None
    assert r['library_mass']==0


@pytest.mark.parametrize('override',[dict(cell_ids=[0,0]),dict(selected_ids=[9]),dict(G_lib=[0,0]),dict(gene_ids=[True])])
def test_bulk_identity_errors(override):
    with pytest.raises(ContractViolation):bulk([[1,2]],**override)


def receipt(ids,h,mass=10):return dict(status='OK',eval_gene_ids=ids,h=h,library_mass=mass)


def test_effect_join_is_by_gene_id():
    r=PairedEffectError(selected_stim=receipt([20,10],[4.,2.]),selected_ctrl=receipt([10,20],[1.,1.]),
                       reference_stim=receipt([10,20],[3.,2.]),reference_ctrl=receipt([20,10],[1.,1.]))
    np.testing.assert_array_equal(r['e_g'],[3,1]);np.testing.assert_array_equal(r['e_ref_g'],[1,2])
    np.testing.assert_array_equal(r['absolute_deviation'],[2,1]);assert r['mean_absolute_error']==1.5
    with pytest.raises(ContractViolation):
        PairedEffectError(selected_stim=receipt([20,10],[4.,2.]),selected_ctrl=receipt([10,21],[1.,1.]),
                          reference_stim=receipt([10,20],[3.,2.]),reference_ctrl=receipt([20,10],[1.,1.]))


def test_effect_propagates_unavailable_status():
    a=bulk([[0]])
    r=PairedEffectError(selected_stim=a,selected_ctrl=a,reference_stim=a,reference_ctrl=a)
    assert r['status']=='NOT_EVALUABLE_ZERO_LIBRARY' and r['mean_absolute_error'] is None


def test_one_vote_per_group_and_reject_repeated_group():
    probs=[[1,0]]+[[0,1]]*7
    r=CloneVoteFateDistribution(probs,group_ids=list(range(8)),selected_group_ids=list(range(8)))
    np.testing.assert_array_equal(r['predicted_probability'],[.125,.875])
    np.testing.assert_array_equal(r['pre_normalization_vote_mass'],np.ones(8))
    with pytest.raises(ContractViolation):CloneVoteFateDistribution(probs,group_ids=list(range(8)),selected_group_ids=[0,0])


def test_soft_reference_and_categorical_brier_are_distinct():
    r=SquaredProbabilityError([.5,.5],reference=[.75,.25],descendant_fate_ids=[0,0,0,1])
    assert r['soft_error']==.125 and r['offset']==.375 and r['mean_categorical_Brier']==.5
    r=SquaredProbabilityError([.75,.25],reference=[.75,.25],descendant_count=4)
    assert r['soft_error']==0 and r['mean_categorical_Brier']==.375


@pytest.mark.parametrize('bad',[[-.1,1.1],[.5,.500001],[float('nan'),0],[True,False]])
def test_invalid_simplex(bad):
    with pytest.raises(ContractViolation):SquaredProbabilityError(bad,reference=[.5,.5])


def test_descendant_reference_mismatch_rejected():
    with pytest.raises(ContractViolation):SquaredProbabilityError([.5,.5],reference=[.5,.5],descendant_fate_ids=[0,0,0,1])


def test_weighted_ids_must_be_aligned_not_just_same_length():
    kwargs=dict(ids=[20,10,30],mass_p=[1,1,0],mass_q=[1,1,98],attribute_range=[0,1])
    assert weighted_total_variation(**kwargs)['TV']==pytest.approx(.98)
    with pytest.raises(ContractViolation):weighted_total_variation(**kwargs,ids_q=[20,10,31])
    with pytest.raises(ContractViolation):weighted_total_variation(**dict(kwargs,ids=[20,10,10]))


def test_zero_mass_and_out_of_range_attribute():
    kwargs=dict(ids=[0,1],mass_p=[0,0],mass_q=[1,1],attribute_range=[0,1])
    r=weighted_total_variation(**kwargs)
    assert r['TV'] is None and r['status']=='NOT_EVALUABLE_ZERO_MASS'
    with pytest.raises(ContractViolation):weighted_total_variation(**kwargs,attribute=[0,2])


def test_mixed_boolean_ids_rejected_before_coercion():
    with pytest.raises(ContractViolation):
        weighted_total_variation(ids=[2,True],mass_p=[1,0],mass_q=[0,1],attribute_range=[0,1])
