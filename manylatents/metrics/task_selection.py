"""Target-blind finite selection policies with explicit stable row identities."""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from collections.abc import Mapping
import numpy as np
from manylatents.metrics.registry import register_metric
from manylatents.utils.task_errors import ContractViolation, NumericalFailure


def _require(ok, message):
    if not ok: raise ContractViolation(message)


def _ids(value, name, unique=True):
    a=np.asarray(value)
    _require(a.ndim==1 and a.dtype==np.dtype('int64'),name+' must be int64 vector')
    _require(not unique or len(np.unique(a))==len(a),name+' must be unique')
    return a.copy()


def _coordinates(value):
    a=np.asarray(value)
    _require(a.ndim==2 and a.shape[1]>0 and a.dtype==np.float64 and np.isfinite(a).all(),'coordinates must be finite float64 (n,d)')
    return a.copy()


def _scalar(value,name,positive=False):
    a=np.asarray(value)
    _require(a.ndim==0 and a.dtype.kind in 'fiu' and np.isfinite(a),name+' must be finite real scalar')
    x=float(a)
    _require(x>0 if positive else x>=0,name+' out of range')
    return x


def _mask(ids, subset, name):
    # Empty Python tuples are the documented default exclusion set.
    values=np.asarray(subset)
    if values.size==0 and values.ndim==1: values=np.array([],dtype=np.int64)
    values=_ids(values,name)
    _require(np.isin(values,ids).all(),name+' contains unknown identity')
    return np.isin(ids,values)


def _labels(value,n,name):
    a=np.asarray(value)
    _require(a.shape==(n,) and a.dtype.kind in 'iuUS',name+' must be aligned integer/string group labels')
    _require(a.dtype.kind!='u' or not a.size or a.max()<=np.iinfo(np.int64).max,name+' out of int64 range')
    return a.copy()


def _json(value):
    if isinstance(value,np.ndarray):return {'dtype':value.dtype.str,'shape':list(value.shape),'values':value.tolist()}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Mapping):
        _require(all(isinstance(k,str) for k in value),'policy mapping keys must be strings')
        return {k:_json(v) for k,v in sorted(value.items())}
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    _require(value is None or isinstance(value,(str,int,float,bool)),'unsupported policy value')
    if isinstance(value,float):_require(np.isfinite(value),'nonfinite policy value')
    return value


def _hash(value):return hashlib.sha256(json.dumps(_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _freeze(value):
    if isinstance(value,np.ndarray):
        # Bytes-backed storage cannot be made writeable again by setflags.
        return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
    if isinstance(value,Mapping):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(_freeze(v) for v in value)
    _json(value)
    return value


@dataclass(frozen=True,init=False)
class SelectionView:
    ids: np.ndarray
    coordinates: np.ndarray
    candidate_mask: np.ndarray
    group_labels: object
    weights: object
    policy_params: object
    hashes: object

    def __init__(self, *, ids, coordinates, candidate_mask, group_labels=None, weights=None, policy_params=None, hashes=None, **unknown):
        _require(not unknown,'SelectionView rejects unknown/forbidden fields: '+','.join(sorted(unknown)))
        x=_coordinates(coordinates);i=_ids(ids,'ids');m=np.asarray(candidate_mask)
        _require(len(x)==len(i) and m.shape==(len(i),) and m.dtype==np.bool_,'view alignment invalid')
        g=None if group_labels is None else _labels(group_labels,len(i),'group_labels')
        w=None if weights is None else _weights(weights,len(i))
        params={} if policy_params is None else dict(policy_params)
        hs={} if hashes is None else dict(hashes)
        _json(params);_json(hs)
        for k,v in dict(ids=i,coordinates=x,candidate_mask=m.copy(),group_labels=g,weights=w,policy_params=params,hashes=hs).items():object.__setattr__(self,k,_freeze(v))


def _weights(value,n):
    a=np.asarray(value)
    _require(a.shape==(n,) and a.dtype.kind in 'fiu' and np.isfinite(a).all() and (a>=0).all(),'weights must be finite nonnegative row-aligned mass')
    return a.astype(np.float64)


def _base(embeddings,center,ids,candidate_ids,exclude_ids=()):
    x=_coordinates(embeddings);i=_ids(ids,'ids');c=np.asarray(center)
    _require(len(x)==len(i) and c.shape==(x.shape[1],) and c.dtype==np.float64 and np.isfinite(c).all(),'center/IDs must align and center be finite float64')
    eligible=_mask(i,candidate_ids,'candidate_ids') & ~_mask(i,exclude_ids,'exclude_ids')
    try:
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            delta=x-c;q=np.sum(delta*delta,axis=1,dtype=np.float64)
    except FloatingPointError as error:raise NumericalFailure('squared distances overflow') from error
    if not np.isfinite(q).all():raise NumericalFailure('squared distances nonfinite')
    return x,i,c.copy(),eligible,q


def _record(x,i,c,eligible,q,tau,weights,groups,policy):
    selected=eligible&(q<=tau);boundary=eligible&(q==tau)
    with np.errstate(over='ignore'):mass=float(weights[selected].sum(dtype=np.float64))
    if not np.isfinite(mass):raise NumericalFailure('selected mass overflow')
    counts={} if groups is None else {v.item():int(np.sum(selected&(groups==v))) for v in np.unique(groups[selected])}
    return dict(selected_ids=np.sort(i[selected]),selected_mask=selected,squared_distances=q,threshold_squared=float(tau),
                boundary_tie_ids=np.sort(i[boundary]),selected_weight_sum=mass,group_counts=counts,status='OK',
                policy_hash=_hash(policy),input_hash=_hash({'coordinates':x,'ids':i,'center':c,'eligible':eligible,'weights':weights,'groups':groups}))


@register_metric(aliases=['centered_selection_ball'])
def CenteredSelectionBall(embeddings, *, center, ids, candidate_ids, metric_form='identity', threshold_squared,
                          exclude_ids=(), candidate_weights=None, candidate_groups=None, dataset=None,module=None,cache=None):
    """Candidate weights/groups, when supplied, align to all input rows, not ID sort order."""
    _require(isinstance(metric_form,str) and metric_form=='identity','only identity metric form is supported')
    tau=_scalar(threshold_squared,'threshold_squared');x,i,c,e,q=_base(embeddings,center,ids,candidate_ids,exclude_ids)
    w=np.ones(len(i)) if candidate_weights is None else _weights(candidate_weights,len(i))
    g=None if candidate_groups is None else _labels(candidate_groups,len(i),'candidate_groups')
    return _record(x,i,c,e,q,tau,w,g,{'name':'CenteredSelectionBall','metric_form':'identity','threshold_squared':tau,'exclude_ids':np.sort(i[_mask(i,exclude_ids,'exclude_ids')])})


def _strata(embeddings,ids,strata,required_strata):
    x=_coordinates(embeddings);i=_ids(ids,'ids');_require(len(x)==len(i),'ID alignment')
    s=_labels(strata,len(i),'strata');r=np.asarray(required_strata)
    _require(r.ndim==1 and len(r)>0 and r.dtype.kind in 'iuUS' and len(np.unique(r))==len(r),'required_strata must be nonempty and unique')
    _require(np.isin(s,r).all(),'undeclared stratum')
    counts=np.array([np.sum(s==v) for v in r],dtype=np.int64)
    return x,i,s,r,counts


@register_metric(aliases=['stratified_seed_centroid'])
def StratifiedSeedCentroid(embeddings, *, ids,strata,required_strata,dataset=None,module=None,cache=None):
    x,i,s,r,counts=_strata(embeddings,ids,strata,required_strata)
    out=dict(status='ABSTAIN_EMPTY_STRATUM',center=None,stratum_centroids=None,seed_weights=None,
             stratum_ids=r.copy(),stratum_counts=counts,policy_hash=_hash({'name':'StratifiedSeedCentroid','required_strata':r}),input_hash=_hash({'coordinates':x,'ids':i,'strata':s}))
    if (counts==0).any():return out
    try:
        with np.errstate(over='raise',invalid='raise'):
            centers=np.stack([np.mean(x[s==v],axis=0) for v in r]);center=np.mean(centers,axis=0)
    except FloatingPointError as error:raise NumericalFailure('centroid overflow') from error
    weights=np.zeros(len(i))
    for v,n in zip(r,counts):weights[s==v]=1./(len(r)*int(n))
    out.update(status='OK',center=center,stratum_centroids=centers,seed_weights=weights)
    return out


@register_metric(aliases=['weighted_coverage_radius'])
def WeightedCoverageRadius(embeddings, *, center,ids,strata,required_strata,rho,dataset=None,module=None,cache=None):
    rho=_scalar(rho,'rho',True);_require(rho<=1,'rho must be <=1')
    strat=StratifiedSeedCentroid(embeddings,ids=ids,strata=strata,required_strata=required_strata)
    x,i,c,e,q=_base(embeddings,center,ids,ids)
    out=dict(status=strat['status'],threshold_squared=None,selected_seed_ids=None,selected_mask=None,selected_seed_mass=None,
             squared_distances=q,seed_weights=strat['seed_weights'],policy_hash=_hash({'name':'WeightedCoverageRadius','rho':rho,'required_strata':np.asarray(required_strata)}),
             input_hash=_hash({'coordinates':x,'ids':i,'center':c,'strata':np.asarray(strata),'weights':strat['seed_weights']}))
    if strat['status']!='OK':return out
    w=strat['seed_weights'];order=np.lexsort((i,q));mass=np.cumsum(w[order],dtype=np.float64)
    # rho=1 means all positive calibration seed mass, even when floating summation is below 1.
    if rho==1:idx=len(order)-1
    else:
        found=np.flatnonzero(mass>=rho)
        if not len(found):raise NumericalFailure('coverage mass cannot reach rho in float64')
        idx=int(found[0])
    tau=float(q[order[idx]]);selected=q<=tau
    out.update(status='OK',threshold_squared=tau,selected_seed_ids=np.sort(i[selected]),selected_mask=selected,selected_seed_mass=float(w[selected].sum(dtype=np.float64)))
    return out


@register_metric(aliases=['distinct_group_budget_ball'])
def DistinctGroupBudgetBall(embeddings, *, center,ids,candidate_ids,group_ids,K_groups,exclude_ids=(),dataset=None,module=None,cache=None):
    _require(isinstance(K_groups,(int,np.integer)) and not isinstance(K_groups,(bool,np.bool_)) and K_groups>0,'K_groups must be positive integer')
    x,i,c,e,q=_base(embeddings,center,ids,candidate_ids,exclude_ids);g=_ids(group_ids,'group_ids',unique=False)
    _require(len(g)==len(i),'group_ids must align to rows')
    groups=np.unique(g[e]);minimum=np.array([q[e&(g==v)].min() for v in groups],dtype=np.float64)
    policy={'name':'DistinctGroupBudgetBall','K_groups':int(K_groups)}
    out=dict(status='ABSTAIN_INSUFFICIENT_GROUPS',selected_ids=None,selected_mask=None,selected_group_ids=None,threshold_squared=None,
             selected_weight_sum=None,group_counts=None,squared_distances=q,boundary_tie_ids=np.array([],dtype=np.int64),
             group_ids=groups,group_min_q=minimum,candidate_threshold_squared=None,candidate_group_count=None,diagnostic_candidate_mask=None,
             noneligible_inside_ids=None,policy_hash=_hash(policy),input_hash=_hash({'coordinates':x,'ids':i,'center':c,'eligible':e,'groups':g}))
    if len(groups)<K_groups:return out
    tau=float(np.sort(minimum)[K_groups-1]);inside=e&(q<=tau);admitted=groups[minimum<=tau]
    out.update(candidate_threshold_squared=tau,candidate_group_count=len(admitted),diagnostic_candidate_mask=inside.copy(),
               boundary_tie_ids=np.sort(i[e&(q==tau)]),noneligible_inside_ids=np.sort(i[~e&(q<=tau)]))
    if len(admitted)>K_groups:out['status']='ABSTAIN_TIE_OVERFLOW';return out
    record=_record(x,i,c,e,q,tau,np.ones(len(i)),g,policy)
    # Keep one stable input hash across success and abstention paths.
    record['input_hash']=out['input_hash'];out.update(record,selected_group_ids=admitted)
    return out


@register_metric(aliases=['donor_eligibility_by_answer_token'])
def DonorEligibilityByAnswerToken(embeddings=None, *, ids,donor_name_pairs,donor_answer_tokens,recipient_pair,recipient_desired_token,dataset=None,module=None,cache=None):
    i=_ids(ids,'ids');answers=_ids(donor_answer_tokens,'donor_answer_tokens',unique=False);pairs=np.asarray(donor_name_pairs)
    pair=_ids(recipient_pair,'recipient_pair');token=np.asarray(recipient_desired_token)
    _require(pairs.shape==(len(i),2) and pairs.dtype==np.int64 and len(answers)==len(i),'donor keys must be aligned int64')
    _require(len(pair)==2 and (pairs[:,0]!=pairs[:,1]).all(),'name pair must contain two distinct tokens')
    _require(token.ndim==0 and token.dtype==np.int64 and int(token)>=0,'recipient token must be nonnegative int64')
    _require((pairs>=0).all() and (answers>=0).all() and (pair>=0).all(),'token IDs must be nonnegative')
    _require(int(token) in pair and np.any(pairs==answers[:,None],axis=1).all(),'answer token must belong to its name pair')
    matchpair=(np.sort(pairs,axis=1)==np.sort(pair)).all(axis=1);matchanswer=answers==token;eligible=matchpair&matchanswer
    reasons=np.where(eligible,'ELIGIBLE',np.where(~matchpair&~matchanswer,'NAME_PAIR_AND_ANSWER_MISMATCH',np.where(~matchpair,'NAME_PAIR_MISMATCH','ANSWER_TOKEN_MISMATCH')))
    return dict(eligible_ids=np.sort(i[eligible]),rejected_ids=np.sort(i[~eligible]),eligible_mask=eligible,rejection_reasons={int(k):str(v) for k,v in zip(i[~eligible],reasons[~eligible])},
                row_reasons=reasons,comparison_keys={'recipient_pair':np.sort(pair),'recipient_desired_token':int(token)},
                name_pair_match=matchpair,answer_token_match=matchanswer,status='OK',
                policy_hash=_hash({'name':'DonorEligibilityByAnswerToken','recipient_pair':np.sort(pair),'recipient_desired_token':int(token)}),
                input_hash=_hash({'ids':i,'donor_name_pairs':pairs,'donor_answer_tokens':answers}))
