"""Registry and Hydra route contracts; no fixture packet or metric calculations."""
import copy
import inspect
from functools import partial
from pathlib import Path
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
import manylatents.metrics as metrics
from manylatents.utils.metrics import flatten_and_unroll_metrics

CASES = [
 ('centered_selection_ball','CenteredSelectionBall','centered_selection_ball'),
 ('stratified_seed_centroid','StratifiedSeedCentroid','stratified_seed_centroid'),
 ('weighted_coverage_radius','WeightedCoverageRadius','weighted_coverage_radius'),
 ('distinct_group_budget_ball','DistinctGroupBudgetBall','distinct_group_budget_ball'),
 ('donor_eligibility_by_answer_token','DonorEligibilityByAnswerToken','donor_eligibility_by_answer_token'),
 ('selected_pseudobulk_log_cpm','SelectedPseudobulkLogCPM','selected_pseudobulk_log_cpm'),
 ('paired_effect_error','PairedEffectError','paired_effect_error'),
 ('clone_vote_fate_distribution','CloneVoteFateDistribution','clone_vote_fate_distribution'),
 ('squared_probability_error','SquaredProbabilityError','squared_probability_error'),
 ('weighted_total_variation','weighted_total_variation','weighted_total_variation'),
 ('paired_logit','PairedLogitContrast','paired_logit_contrast'),
]

@pytest.mark.parametrize('config_name,name,alias',CASES)
def test_export_registry_alias_and_standard_hydra_route(config_name,name,alias):
    function=getattr(metrics,name)
    assert name in metrics.__all__
    assert metrics.get_metric(name).func is function
    assert metrics.get_metric(alias).func is function
    assert metrics.resolve_metric(alias)==(function,{})
    parameters=inspect.signature(function).parameters
    assert {'embeddings','dataset','module','cache'} <= parameters.keys()
    path=Path(metrics.__file__).resolve().parents[1]/'configs'/'metrics'/(config_name+'.yaml')
    cfg=OmegaConf.load(path)
    # Exercise the actual group flattening route: flat target files are silently
    # discarded; mandatory ??? fields raise during iteration; arrays would sweep.
    flattened=flatten_and_unroll_metrics(cfg)
    assert list(flattened)==[config_name]
    node=copy.deepcopy(flattened[config_name])
    assert node.at=='embedding'
    del node.at
    caller=instantiate(node)
    assert isinstance(caller,partial) and caller.func is function
    assert not caller.args
    assert not any(isinstance(v,(list,tuple)) for v in caller.keywords.values())
    # Binding the runtime signature verifies config kwargs without executing it.
    inspect.signature(function).bind_partial(**caller.keywords)


def test_non_string_dtype_contract_rejected_before_hook_installation():
    import torch
    from manylatents.utils.causal_activation import capture_context
    from manylatents.utils.task_errors import ContractViolation
    class NeverForward(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.block=torch.nn.Identity()
        def forward(self,**kwargs):
            raise AssertionError('Invalid contract must fail before forward')
    model=NeverForward().eval()
    contract=dict(site_path='block',output_kind='tensor',width=2,dtype=32,
                  device='cpu',token_position_policy='explicit_nonpadding')
    with pytest.raises(ContractViolation):
        capture_context(model,torch.tensor([[0]]),torch.tensor([[1]]),site_path='block',
                        site_contract=contract,token_positions=torch.tensor([0]))
    assert not model.block._forward_hooks
