"""Native ID-aligned task summaries, without biological or causal guarantees."""
from __future__ import annotations
import math
import numpy as np
from scipy import sparse
from manylatents.metrics.registry import register_metric
from manylatents.utils.task_errors import ContractViolation, NumericalFailure

_MAX_INT64 = np.iinfo(np.int64).max


def _ids(value, name, length=None):
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ContractViolation(f'{name} cannot contain boolean IDs')
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and (raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b')):
        raise ContractViolation(f'{name} must contain integer IDs')
    vals = [int(v) for v in raw]
    if any(v < np.iinfo(np.int64).min or v > _MAX_INT64 for v in vals):
        raise ContractViolation(f'{name} is outside int64 range')
    if len(set(vals)) != len(vals) or (length is not None and len(vals) != length):
        raise ContractViolation(f'{name} must be unique and aligned')
    return np.asarray(vals, dtype=np.int64)


def _real(value, name, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ContractViolation(f'{name} must be finite real numeric with ndim={ndim}')
    a = a.astype(np.float64)
    if not np.isfinite(a).all(): raise NumericalFailure(f'{name} float64 conversion overflow')
    return a


def _simplex(value, name, ndim):
    a = _real(value, name, ndim)
    if not a.shape[-1] or (a < 0).any(): raise ContractViolation(f'{name} is not a simplex')
    with np.errstate(over='ignore', invalid='ignore'):
        total = a.sum(axis=-1)
    if not np.isfinite(total).all() or np.any(np.abs(total-1) > 1e-12):
        raise ContractViolation(f'{name} simplex mass must equal one within 1e-12')
    return a


def _subset(subset, universe, name):
    ids = _ids(subset, name)
    mapping = {int(v): i for i,v in enumerate(universe)}
    if any(int(v) not in mapping for v in ids): raise ContractViolation(f'{name} has unknown IDs')
    return ids, np.array([mapping[int(v)] for v in ids], dtype=np.int64)


@register_metric(aliases=['selected_pseudobulk_log_cpm'])
def SelectedPseudobulkLogCPM(embeddings, *, cell_ids, gene_ids, selected_ids,
                             G_lib, G_eval, min_cells=1, dataset=None, module=None, cache=None):
    """Exact sparse integer aggregation with a frozen denominator gene universe.

    Sparse entries are validated before any reduction or float conversion. Python
    integer accumulation avoids wraparound, with explicit int64 output bounds.
    No dense genes-by-cells matrix is ever constructed. Duplicate sparse entries
    contribute their nonnegative counts independently, as sparse semantics imply.
    """
    if not sparse.issparse(embeddings) or embeddings.ndim != 2:
        raise ContractViolation('embeddings must be a sparse genes-by-cells count matrix')
    if isinstance(min_cells, (bool,np.bool_)) or not isinstance(min_cells,(int,np.integer)) or min_cells<1:
        raise ContractViolation('min_cells must be a positive integer')
    n_genes,n_cells=embeddings.shape
    cells=_ids(cell_ids,'cell_ids',n_cells); genes=_ids(gene_ids,'gene_ids',n_genes)
    selected,cols=_subset(selected_ids,cells,'selected_ids')
    lib,lib_rows=_subset(G_lib,genes,'G_lib'); evaluation,eval_rows=_subset(G_eval,genes,'G_eval')
    if not len(lib) or not len(evaluation): raise ContractViolation('Gene universes must be nonempty')
    coo=embeddings.tocoo(copy=False)
    if coo.data.dtype.kind not in 'iuf': raise ContractViolation('Counts cannot be boolean, complex or objects')
    # Validate even entries outside the selected subset. Do not silently excuse
    # malformed count input when a coverage status would otherwise abstain.
    for v in coo.data:
        if not np.isfinite(v) or v<0 or (coo.data.dtype.kind=='f' and v!=np.floor(v)):
            raise ContractViolation('Counts must be finite nonnegative mathematical integers')
        if int(v)>_MAX_INT64: raise NumericalFailure('Count exceeds int64 range')
    base=dict(selected_ids=selected.copy(),eval_gene_ids=evaluation.copy(),gene_ids=genes.copy(),
              denominator_gene_ids=lib.copy(),n_selected=len(selected),min_cells=int(min_cells))
    if len(selected)<min_cells:
        return dict(base,status='NOT_EVALUABLE_MIN_CELLS',gene_sums=None,library_mass=None,h=None)
    selected_cols=set(int(c) for c in cols); sums=[0]*n_genes
    for r,c,v in zip(coo.row,coo.col,coo.data):
        if int(c) in selected_cols:
            sums[int(r)]+=int(v)
            if sums[int(r)]>_MAX_INT64: raise NumericalFailure('Per-gene int64 accumulation overflow')
    mass=sum(sums[int(r)] for r in lib_rows)
    if mass>_MAX_INT64: raise NumericalFailure('Library int64 accumulation overflow')
    gene_sums=np.asarray(sums,dtype=np.int64)
    if mass==0:return dict(base,status='NOT_EVALUABLE_ZERO_LIBRARY',gene_sums=gene_sums,library_mass=0,h=None)
    h=np.array([math.log2(1+1e6*sums[int(r)]/mass) for r in eval_rows],dtype=np.float64)
    if not np.isfinite(h).all(): raise NumericalFailure('LogCPM nonfinite output')
    return dict(base,status='OK',gene_sums=gene_sums,library_mass=mass,h=h)


@register_metric(aliases=['paired_effect_error'])
def PairedEffectError(embeddings=None, *, selected_stim, selected_ctrl,
                      reference_stim, reference_ctrl, dataset=None, module=None, cache=None):
    """Compare paired logCPM effects using ID joins, never positional coincidence."""
    receipts=[selected_stim,selected_ctrl,reference_stim,reference_ctrl]
    names=['selected_stim','selected_ctrl','reference_stim','reference_ctrl']
    allowed={'OK','NOT_EVALUABLE_MIN_CELLS','NOT_EVALUABLE_ZERO_LIBRARY'}
    ids=[_ids(r['eval_gene_ids'],name+'.eval_gene_ids') for name,r in zip(names,receipts)]
    if not len(ids[0]) or any(set(a.tolist())!=set(ids[0].tolist()) for a in ids[1:]):
        raise ContractViolation('Evaluation gene identities differ across paired effects')
    if any(r['status'] not in allowed for r in receipts):raise ContractViolation('Unknown pseudobulk receipt status')
    masses={name:r['library_mass'] for name,r in zip(names,receipts)}
    status=next((r['status'] for r in receipts if r['status']!='OK'),'OK')
    if status!='OK':
        return dict(status=status,eval_gene_ids=ids[0].copy(),e_g=None,e_ref_g=None,
                    absolute_deviation=None,mean_absolute_error=None,library_masses=masses)
    values=[]
    for name,r,gene_map in zip(names,receipts,ids):
        h=_real(r['h'],name+'.h',1)
        if len(h)!=len(gene_map):raise ContractViolation('LogCPM values and gene IDs differ in length')
        positions={int(g):i for i,g in enumerate(gene_map)}
        values.append(np.asarray([h[positions[int(g)]] for g in ids[0]]))
    with np.errstate(over='ignore',invalid='ignore'):
        effect=values[0]-values[1]; ref=values[2]-values[3]; error=np.abs(effect-ref)
    if not np.isfinite(effect).all() or not np.isfinite(ref).all() or not np.isfinite(error).all():
        raise NumericalFailure('Paired effect arithmetic overflow')
    mean=float(np.mean(error))
    if not math.isfinite(mean):raise NumericalFailure('Paired effect mean overflow')
    return dict(status='OK',eval_gene_ids=ids[0].copy(),e_g=effect,e_ref_g=ref,
                absolute_deviation=error,mean_absolute_error=mean,library_masses=masses)


@register_metric(aliases=['clone_vote_fate_distribution'])
def CloneVoteFateDistribution(embeddings, *, group_ids, selected_group_ids,
                              dataset=None,module=None,cache=None):
    """One vote per unique selected group, independent of its cell multiplicity."""
    probabilities=_simplex(embeddings,'training_outcomes',2)
    groups=_ids(group_ids,'group_ids',len(probabilities))
    chosen,rows=_subset(selected_group_ids,groups,'selected_group_ids')
    if not len(chosen):
        return dict(status='NOT_EVALUABLE_EMPTY_SELECTION',selected_group_ids=chosen,
                    pre_normalization_vote_mass=None,predicted_probability=None)
    p=probabilities[rows].mean(axis=0)
    return dict(status='OK',selected_group_ids=chosen.copy(),
                pre_normalization_vote_mass=np.ones(len(chosen),dtype=np.float64),predicted_probability=p)


@register_metric(aliases=['squared_probability_error'])
def SquaredProbabilityError(embeddings, *, reference, descendant_count=None,
                            descendant_fate_ids=None,dataset=None,module=None,cache=None):
    """Squared error to a soft reference, separate from categorical Brier offset."""
    p=_simplex(embeddings,'prediction',1);o=_simplex(reference,'reference',1)
    if p.shape!=o.shape:raise ContractViolation('Prediction and reference vocabularies have different lengths')
    if descendant_count is not None and (isinstance(descendant_count,(bool,np.bool_)) or
            not isinstance(descendant_count,(int,np.integer)) or descendant_count<0):
        raise ContractViolation('descendant_count must be a nonnegative integer')
    if descendant_fate_ids is not None:
        raw=np.asarray(descendant_fate_ids)
        if raw.ndim!=1 or (raw.size and raw.dtype.kind not in 'iu'):
            raise ContractViolation('Descendant fate IDs must be integer vocabulary positions')
        if ((raw<0)|(raw>=len(o))).any():raise ContractViolation('Unknown descendant fate ID')
        if descendant_count is not None and descendant_count!=len(raw):raise ContractViolation('Descendant count mismatch')
        descendant_count=len(raw)
        if descendant_count:
            proportions=np.array([np.count_nonzero(raw==i)/descendant_count for i in range(len(o))])
            if np.any(np.abs(proportions-o)>1e-12):raise ContractViolation('Soft reference does not match descendant labels')
    error=float(np.sum((p-o)**2));offset=float(1-np.sum(o**2))
    return dict(status='OK',soft_error=error,offset=offset,mean_categorical_Brier=error+offset,
                descendant_count=None if descendant_count is None else int(descendant_count),
                prediction=p.copy(),reference=o.copy())


@register_metric(aliases=['weighted_total_variation'])
def weighted_total_variation(embeddings=None, *, ids,mass_p,mass_q,attribute_range,
                              ids_p=None,ids_q=None,attribute=None,dataset=None,module=None,cache=None):
    """TV on an explicit ordered ID union; normalization and weights are exposed."""
    union=_ids(ids,'ids')
    for name,side in [('ids_p',ids_p),('ids_q',ids_q)]:
        if side is not None and not np.array_equal(_ids(side,name),union):
            raise ContractViolation('Mass vectors must use the same explicit ordered ID union')
    p=_real(mass_p,'mass_p',1);q=_real(mass_q,'mass_q',1)
    if len(p)!=len(union) or len(q)!=len(union) or (p<0).any() or (q<0).any():
        raise ContractViolation('Mass vectors must be nonnegative and ID aligned')
    bounds=_real(attribute_range,'attribute_range',1)
    if bounds.shape!=(2,) or bounds[1]<bounds[0]:raise ContractViolation('Invalid attribute range')
    f=None if attribute is None else _real(attribute,'attribute',1)
    if f is not None and (f.shape!=p.shape or (f<bounds[0]).any() or (f>bounds[1]).any()):
        raise ContractViolation('Attribute must align with IDs and lie in declared range')
    with np.errstate(over='ignore',invalid='ignore'):
        total_p=float(p.sum());total_q=float(q.sum());width=float(bounds[1]-bounds[0])
    if not all(math.isfinite(x) for x in (total_p,total_q,width)):
        raise NumericalFailure('Mass or range sum overflow')
    base=dict(ids=union.copy(),total_mass_p=total_p,total_mass_q=total_q,attribute_range=bounds.copy())
    if total_p==0 or total_q==0:
        return dict(base,status='NOT_EVALUABLE_ZERO_MASS',TV=None,bounded_attribute_product=None,
                    normalized_p=None,normalized_q=None,observed_weighted_attribute_error=None)
    p=p/total_p;q=q/total_q;tv=float(.5*np.abs(p-q).sum());product=width*tv
    # Center f before integration: subtraction of a common offset leaves the
    # mean difference unchanged and avoids needless cancellation of large means.
    observed=None if f is None else abs(float(np.dot(p-q,f-bounds[0])))
    if not all(math.isfinite(v) for v in [tv,product]+([] if observed is None else [observed])):
        raise NumericalFailure('Weighted summary arithmetic overflow')
    return dict(base,status='OK',TV=tv,bounded_attribute_product=product,normalized_p=p,
                normalized_q=q,observed_weighted_attribute_error=observed)
