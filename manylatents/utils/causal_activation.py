"""Explicit single-site activation capture/replacement; never loads a model."""
from __future__ import annotations
import hashlib
from .task_errors import ContractViolation, NumericalFailure


def _require(ok,message):
    if not bool(ok): raise ContractViolation(message)


def _finite(tensor,name):
    import torch
    if not bool(torch.isfinite(tensor).all()): raise NumericalFailure(name+' contains nonfinite values')


def _prepare(model,input_ids,attention_mask,site_path,site_contract,token_positions,use_cache,past_key_values):
    import torch
    _require(use_cache is False and past_key_values is None,'Cache use is forbidden')
    _require(isinstance(site_contract,dict),'site_contract must be explicit')
    needed={'site_path','output_kind','width','dtype','device','token_position_policy'}
    _require(needed.issubset(site_contract),'Incomplete site_contract')
    _require(site_contract['site_path']==site_path,'Resolved site path differs')
    _require(site_contract['output_kind'] in ('tensor','tuple'),'Unsupported declared output kind')
    _require(site_contract['token_position_policy']=='explicit_nonpadding','Unsupported position policy')
    width=site_contract['width']
    _require(isinstance(width,int) and not isinstance(width,bool) and width>0,'Invalid width')
    _require(isinstance(site_contract['dtype'],str),'dtype contract must be a string')
    dtype=getattr(torch,site_contract['dtype'],None)
    _require(dtype in (torch.float16,torch.float32,torch.float64,torch.bfloat16),'Unsupported floating dtype')
    try: device=torch.device(site_contract['device'])
    except (RuntimeError,TypeError) as exc: raise ContractViolation('Invalid device') from exc
    _require(hasattr(model,'modules') and not any(m.training for m in model.modules()),'Model must be in evaluation mode')
    try: site=model.get_submodule(site_path)
    except (AttributeError,KeyError,TypeError) as exc: raise ContractViolation('Site path does not resolve') from exc
    _require(isinstance(input_ids,torch.Tensor) and input_ids.dtype in (torch.int32,torch.int64),'input_ids must be integer tensor')
    _require(input_ids.ndim==2 and min(input_ids.shape)>0 and bool((input_ids>=0).all()),'Invalid input IDs')
    _require(isinstance(attention_mask,torch.Tensor) and attention_mask.shape==input_ids.shape,'Explicit mask must align with IDs')
    _require(attention_mask.dtype in (torch.bool,torch.int32,torch.int64),'Mask must be boolean/integer')
    _require(bool(((attention_mask==0)|(attention_mask==1)).all()),'Mask must be binary')
    # Right-padding only; do not infer positional numbering for left padding/gaps.
    _require(bool((attention_mask[:,0]==1).all()) and not bool((attention_mask[:,1:]>attention_mask[:,:-1]).any()),'Only contiguous right-padding masks supported')
    _require(isinstance(token_positions,torch.Tensor) and token_positions.dtype in (torch.int32,torch.int64),'Explicit integer token_positions required')
    batch,length=input_ids.shape
    _require(token_positions.shape==(batch,),'Position vector must have shape(batch,)')
    _require(bool(((token_positions>=0)&(token_positions<length)).all()),'Position out of range')
    _require(input_ids.device==device and attention_mask.device==device and token_positions.device==device,'Input device differs from contract')
    row=torch.arange(batch,device=device)
    _require(bool((attention_mask[row,token_positions]==1).all()),'Position points into padding')
    h=hashlib.sha256()
    for tensor in (input_ids,attention_mask,token_positions):
        t=tensor.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
    return site,dtype,device,row,h.hexdigest()


def _run(model,input_ids,attention_mask,*,site_path,site_contract,token_positions,use_cache,past_key_values,donor_vectors=None):
    import torch
    site,dtype,device,row,input_hash=_prepare(model,input_ids,attention_mask,site_path,site_contract,token_positions,use_cache,past_key_values)
    batch,length=input_ids.shape;width=site_contract['width']
    if donor_vectors is not None:
        _require(isinstance(donor_vectors,torch.Tensor) and donor_vectors.shape==(batch,width),'One whole donor vector per batch row required')
        _require(donor_vectors.dtype==dtype and donor_vectors.device==device,'Donor dtype/device differs')
        _finite(donor_vectors,'Donor')
    cache={};calls=0
    def hook(module,args,output):
        nonlocal calls
        calls+=1
        _require(calls==1,'Bound site must execute exactly once')
        if site_contract['output_kind']=='tensor':
            _require(isinstance(output,torch.Tensor),'Runtime output kind differs')
            hidden=output
        else:
            _require(isinstance(output,tuple) and len(output)>0 and isinstance(output[0],torch.Tensor),'Runtime tuple kind differs')
            hidden=output[0]
        _require(hidden.shape==(batch,length,width),'Runtime block dimensions differ')
        _require(hidden.dtype==dtype and hidden.device==device,'Runtime block dtype/device differs')
        _finite(hidden,'Block output')
        cache['block_output']=hidden[row,token_positions].detach().clone()
        if donor_vectors is None:return None
        replaced=hidden.clone()
        replaced[row,token_positions]=donor_vectors
        cache['patched_values']=replaced[row,token_positions].detach().clone()
        return replaced if site_contract['output_kind']=='tensor' else (replaced,)+output[1:]
    handle=site.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,past_key_values=None)
            _require(calls==1,'Bound site was not executed')
            logits=getattr(output,'logits',None)
            _require(isinstance(logits,torch.Tensor) and logits.ndim==3 and logits.shape[:2]==(batch,length),'Model must return complete sequence .logits')
            _require(logits.is_floating_point() and logits.device==device,'Invalid model logits dtype/device')
            _finite(logits,'Model logits')
            cache['logits']=logits[row,token_positions].detach().clone()
    finally:
        handle.remove()
    # Outside inference mode: independent, ordinary detached tensors for callers.
    result={key:value.detach().clone() for key,value in cache.items()}
    result.update(input_hash=input_hash,site_contract=dict(site_contract),token_positions=token_positions.detach().clone())
    return result


def capture_context(model,input_ids,attention_mask,*,site_path,site_contract,token_positions=None,use_cache=False,past_key_values=None):
    """Capture copied vectors and model-readout logits at explicit token positions."""
    return _run(model,input_ids,attention_mask,site_path=site_path,site_contract=site_contract,
                token_positions=token_positions,use_cache=use_cache,past_key_values=past_key_values)


def replace_context_activation(model,input_ids,attention_mask,donor_vectors,*,site_path,site_contract,token_positions=None,use_cache=False,past_key_values=None):
    """Replace one complete vector per recipient; preserve other tokens/tuple tails."""
    result=_run(model,input_ids,attention_mask,site_path=site_path,site_contract=site_contract,
                token_positions=token_positions,use_cache=use_cache,past_key_values=past_key_values,donor_vectors=donor_vectors)
    return dict(logits=result['logits'],patched_values=result['patched_values'],
                patch_metadata={key:result[key] for key in ('input_hash','site_contract','token_positions')})
