"""Structured raw logit contrasts, with no normalization or hidden forward pass."""
import numpy as np
from manylatents.metrics.registry import register_metric
from manylatents.utils.task_errors import ContractViolation, NumericalFailure


def _logits(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu()
        if str(value.dtype)=='torch.bfloat16':value=value.float()
        value=value.numpy()
    value=np.asarray(value)
    if np.iscomplexobj(value) or not np.issubdtype(value.dtype,np.number) or value.ndim!=2 or min(value.shape)<1:
        raise ContractViolation('Logits must be nonempty real numeric(batch,vocab)')
    result=np.asarray(value,dtype=np.float64)
    if not np.isfinite(result).all():raise NumericalFailure('Logits contain nonfinite values')
    return result


def _ids(value,n,vocab):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    value=np.asarray(value)
    if value.shape!=(n,) or not np.issubdtype(value.dtype,np.integer) or (value<0).any() or (value>=vocab).any():
        raise ContractViolation('Token IDs must be valid integer vectors(batch,)')
    return value


@register_metric(aliases=['paired_logit_contrast'])
def PairedLogitContrast(embeddings,*,desired_token_ids,competitor_token_ids,baseline_logits=None,dataset=None,module=None,cache=None):
    """Return logit(desired)-logit(competitor), optionally minus baseline contrast."""
    x=_logits(embeddings);n,vocab=x.shape
    wanted=_ids(desired_token_ids,n,vocab);other=_ids(competitor_token_ids,n,vocab)
    if np.any(wanted==other):raise ContractViolation('Desired and competitor token IDs must differ')
    row=np.arange(n)
    with np.errstate(over='raise',invalid='raise'):
        try:
            contrast=x[row,wanted]-x[row,other]
            baseline=None;effect=None
            if baseline_logits is not None:
                b=_logits(baseline_logits)
                if b.shape!=x.shape:raise ContractViolation('Baseline logits must match shape')
                baseline=b[row,wanted]-b[row,other];effect=contrast-baseline
        except FloatingPointError as exc:raise NumericalFailure('Logit difference overflow') from exc
    return dict(desired_logit=x[row,wanted].copy(),competitor_logit=x[row,other].copy(),contrast=contrast,
                baseline_contrast=baseline,effect=effect,status='OK',units='logits')
