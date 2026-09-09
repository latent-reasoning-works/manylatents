from types import SimpleNamespace
import numpy as np
import pytest
import torch
from manylatents.utils.causal_activation import capture_context,replace_context_activation
from manylatents.metrics.paired_logit import PairedLogitContrast
from manylatents.utils.task_errors import ContractViolation,NumericalFailure

class Toy(torch.nn.Module):
    def __init__(self,kind='tensor'):
        super().__init__();self.kind=kind;self.block=torch.nn.Identity();self.aux=object();self.fail=False
        self.buffer=torch.tensor([[[9.,9.],[1.,2.]]]);self.head=torch.tensor([[1.,2.],[2.,-1.]])
        self.block.register_forward_pre_hook(self.prepare)
    def prepare(self,module,args):return args
    def forward(self,input_ids,attention_mask,use_cache,past_key_values):
        assert use_cache is False and past_key_values is None
        value=self.buffer if self.kind=='tensor' else (self.buffer,self.aux,[17,19])
        out=self.block(value)
        self.last=out
        if self.fail:raise RuntimeError('deliberate forward failure')
        hidden=out if self.kind=='tensor' else out[0]
        return SimpleNamespace(logits=hidden@self.head.T)

def args(kind='tensor'):
    return dict(site_path='block',site_contract=dict(site_path='block',output_kind=kind,width=2,dtype='float32',device='cpu',token_position_policy='explicit_nonpadding'),token_positions=torch.tensor([1]))

def inputs():return torch.tensor([[0,1]]),torch.tensor([[1,1]])

def test_two_donors_tensor_arithmetic_identity_and_clones():
    model=Toy().eval();ids,mask=inputs();kw=args()
    captured=capture_context(model,ids,mask,**kw)
    assert torch.equal(captured['logits'],torch.tensor([[5.,0.]]))
    captured['block_output'][0,0]=99
    assert model.buffer[0,1,0]==1
    effects=[]
    for donor,expected in [([3.,2.],[7.,4.]),([1.,4.],[9.,-2.])]:
        d=torch.tensor([donor]);before=d.clone()
        out=replace_context_activation(model,ids,mask,d,**kw)
        assert torch.equal(out['logits'],torch.tensor([expected]))
        assert torch.equal(model.last[0,0],torch.tensor([9.,9.]))
        assert torch.equal(d,before) and torch.equal(model.buffer[0,1],torch.tensor([1.,2.]))
        effects.extend(PairedLogitContrast(out['logits'],desired_token_ids=[0],competitor_token_ids=[1],baseline_logits=captured['logits'])['effect'])
    assert effects==[-2.,6.]
    identity=replace_context_activation(model,ids,mask,model.buffer[:,1].clone(),**kw)
    assert torch.equal(identity['logits'],captured['logits'])
    assert len(model.block._forward_hooks)==0


def test_tuple_tail_identity_and_order():
    model=Toy('tuple').eval();ids,mask=inputs()
    out=replace_context_activation(model,ids,mask,torch.tensor([[3.,2.]]),**args('tuple'))
    assert model.last[1] is model.aux and model.last[2]==[17,19]
    assert torch.equal(out['logits'],torch.tensor([[7.,4.]]))


def test_positions_cache_kind_and_training_contracts():
    model=Toy().eval();ids,mask=inputs()
    for override in ({'token_positions':None},{'token_positions':torch.tensor([2])},{'use_cache':True},{'past_key_values':object()}):
        kw=args();kw.update(override)
        with pytest.raises(ContractViolation):capture_context(model,ids,mask,**kw)
        assert not model.block._forward_hooks
    padding=torch.tensor([[1,0]])
    with pytest.raises(ContractViolation):capture_context(model,ids,padding,**args())
    kw=args();kw['token_positions']=torch.tensor([0])
    assert capture_context(model,ids,padding,**kw)['block_output'].tolist()==[[9.,9.]]
    with pytest.raises(ContractViolation):capture_context(model,ids,mask,**args('tuple'))
    assert not model.block._forward_hooks
    model.train()
    with pytest.raises(ContractViolation):capture_context(model,ids,mask,**args())


def test_cleanup_and_next_call_after_runtime_error():
    model=Toy().eval();ids,mask=inputs();model.fail=True
    with pytest.raises(RuntimeError,match='deliberate'):replace_context_activation(model,ids,mask,torch.tensor([[3.,2.]]),**args())
    assert not model.block._forward_hooks
    model.fail=False
    assert capture_context(model,ids,mask,**args())['logits'].tolist()==[[5.,0.]]


def test_logit_precision_shape_ids_and_nonfinite():
    x=torch.tensor([[1.+2**-40,1.]],dtype=torch.float64)
    r=PairedLogitContrast(x,desired_token_ids=[0],competitor_token_ids=[1])
    assert r['contrast'][0]==2**-40 and r['effect'] is None
    for desired,competitor in [([0],[0]),([2],[1]),([0.],[1]),([True],[1])]:
        with pytest.raises(ContractViolation):PairedLogitContrast([[1.,0.]],desired_token_ids=desired,competitor_token_ids=competitor)
    with pytest.raises(ContractViolation):PairedLogitContrast(torch.tensor([[1j,0j]]),desired_token_ids=[0],competitor_token_ids=[1])
    with pytest.raises(NumericalFailure):PairedLogitContrast([[np.nan,0]],desired_token_ids=[0],competitor_token_ids=[1])


def test_donor_dimensions_dtype_and_nonfinite():
    model=Toy().eval();ids,mask=inputs()
    for donor in (torch.ones((2,2)),torch.ones((1,2),dtype=torch.float64)):
        with pytest.raises(ContractViolation):replace_context_activation(model,ids,mask,donor,**args())
    with pytest.raises(NumericalFailure):replace_context_activation(model,ids,mask,torch.tensor([[float('nan'),0.]]),**args())
    assert not model.block._forward_hooks
