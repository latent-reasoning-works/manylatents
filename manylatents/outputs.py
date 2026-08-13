"""Registry of GENERIC algorithm outputs — the things the engine can pull off any
fitted algorithm without knowing what it is.

Follows ``metrics/registry.py``: a plain function, a decorator, metadata at the
registration site, enumerable by name, no base class. The point is the last part.
This collect used to be the body of ``LatentModule.extra_outputs()``, an inherited
default, which meant it reached exactly the classes that inherited it. MIOFlow
computes ``.trajectories`` (mioflow.py:364) and could never emit them, because it is
a LightningModule — that is #295, and it is not a bug you can fix by editing MIOFlow,
it is what "generic behaviour lives on a base class" costs. An extractor is generic
over anything that stores the attribute, so the same defect is not expressible here.

Algorithm-SPECIFIC work stays in ``extra_outputs()``: Cflows integrates a trajectory,
decodes it to gene space and runs a Granger estimator (cflows.py:393) — that is a
computation, not an attribute read. ``collect_outputs()`` merges both halves, and it
is the only thing the two engine merge points (experiment.py, evaluate.py) call.

Hard constraint: this module imports nothing from ``manylatents``.
``latent_module_base`` imports it and ``metrics/registry`` imports
``latent_module_base``; anything else here is an import cycle.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

#: What "this algorithm does not have that" looks like, as opposed to "this algorithm
#: is broken". Exactly the tuple the inherited collect used at
#: latent_module_base.py:238 — NotImplementedError is the ABC declining, AttributeError
#: is a hook built out of a missing attribute, RuntimeError is "not fitted yet".
_ABSENT = (NotImplementedError, AttributeError, RuntimeError)


@dataclass
class OutputSpec:
    """A registered extractor plus the metadata that makes it answerable offline."""

    func: Callable[[Any], Dict[str, Any]]
    description: str = ""
    #: Attribute/method names the extractor reads. Metadata only — the extractor still
    #: guards itself — but it makes "what could this algorithm produce" answerable from
    #: the registry WITHOUT fitting anything.
    requires: tuple[str, ...] = field(default_factory=tuple)


_OUTPUT_REGISTRY: Dict[str, OutputSpec] = {}


def register_output(description: str = "", requires: tuple[str, ...] = ()):
    """Register a generic output extractor under its own function name."""

    def decorator(fn: Callable[[Any], Dict[str, Any]]):
        _OUTPUT_REGISTRY[fn.__name__] = OutputSpec(
            fn, description or (fn.__doc__ or ""), requires
        )
        return fn

    return decorator


def get_output_registry() -> Dict[str, OutputSpec]:
    """A copy of the registry — callers enumerate it, they do not own it."""
    return _OUTPUT_REGISTRY.copy()


def list_outputs() -> list[str]:
    """Names of every generic output an algorithm could emit."""
    return sorted(_OUTPUT_REGISTRY)


def generic_default(fn):
    """Mark a method as 'this is only the registry pass'.

    ``collect_outputs()`` skips a marked ``extra_outputs``, because running it would
    recompute every extractor a second time. Measured on ``PCAModule`` at N=3000: one
    generic collect is 10.0 ms and an NxN allocation, and ``kernel()`` is already
    computed 4x per ``run_experiment`` — the two merge points collect once each, and
    ``PCAModule.affinity()`` calls ``kernel()`` internally. The duplication is
    pre-existing; doubling it again for an identical result is not.
    """
    fn._manylatents_registry_default = True
    return fn


# ---------------------------------------------------------------- extractors --


def _accessor(algorithm, name: str):
    """The zero-arg accessor called ``name``, or None if this object has no such thing.

    Guards by PRECONDITION rather than by a blanket ``except``: an ``except Exception``
    around the call would also swallow a genuine bug inside a real ``affinity()`` and
    record a broken output as an absent one. The three preconditions are each a real
    object in this tree or an obvious next one:
      * not callable — someone stores an array under a hook name.
      * needs an argument — ``gpu_phate_local.PHATE`` sets ``self.affinity =
        PHATEAffinity(...)`` (gpu_phate_local.py:367), a callable instance taking X.
      * an ``nn.Module`` — a LAYER named ``kernel`` on a LightningModule. Calling it
        would run a forward pass, which is not reading an output.
    """
    # THE READ IS GUARDED, THE CALL IS NOT, and the asymmetry is the point. A hook may be a
    # `@property`, and a property that raises "not fitted" on access is stating an ABSENT output
    # — which is how `LatentModule.extra_outputs` treated it before this module existed, because
    # its `getattr` sat INSIDE the try at latent_module_base.py:238. Leaving the read bare here
    # let that exception escape `collect_outputs()` and abort a run that previously completed.
    # Narrow, and the same three the extractors swallow: a genuine bug inside a real `affinity()`
    # still propagates, because that happens at CALL time, below.
    try:
        fn = getattr(algorithm, name, None)
    except (NotImplementedError, AttributeError, RuntimeError):
        return None
    if not callable(fn) or isinstance(fn, torch.nn.Module):
        return None
    try:
        inspect.signature(fn).bind()
    except (TypeError, ValueError):
        return None
    return fn


def _matrix_output(algorithm, name: str) -> dict:
    fn = _accessor(algorithm, name)
    if fn is None:
        return {}
    try:
        return {name: fn()}
    except _ABSENT:
        return {}


@register_output(
    description="Flow paths through the latent space, (n_bins, n_trajectories, d)",
    requires=("trajectories",),
)
def trajectories(algorithm) -> dict:
    """Integrated trajectories, stored as an attribute rather than a method."""
    try:
        traj = getattr(algorithm, "trajectories", None)
    except _ABSENT:
        # A property that raises before fit. `getattr(obj, name, default)` only rescues
        # AttributeError, so an unfitted property would otherwise escape from here while
        # an unfitted `affinity()` was being swallowed two lines down.
        return {}
    if traj is None:
        return {}
    return {"trajectories": traj.detach().cpu().numpy() if isinstance(traj, Tensor) else traj}


@register_output(
    description="Normalised transition matrix / diffusion operator, (N, N)",
    requires=("affinity",),
)
def affinity(algorithm) -> dict:
    """Affinity matrix, as the algorithm's ``affinity()`` returns it."""
    return _matrix_output(algorithm, "affinity")


@register_output(
    description="Binary graph connectivity, (M, M) — M need not be N",
    requires=("adjacency",),
)
def adjacency(algorithm) -> dict:
    """Adjacency matrix, as the algorithm's ``adjacency()`` returns it."""
    return _matrix_output(algorithm, "adjacency")


@register_output(
    description="Raw similarity / Gram matrix, (N, N)",
    requires=("kernel",),
)
def kernel(algorithm) -> dict:
    """Kernel matrix, as the algorithm's ``kernel()`` returns it."""
    return _matrix_output(algorithm, "kernel")


# ------------------------------------------------------------------- collect --


def collect_registered_outputs(algorithm) -> dict:
    """Run every registered extractor against ``algorithm``.

    Deliberately no blanket ``except`` around the extractors: each one already declines
    on precondition and swallows only `_ABSENT`, so anything that escapes is a defect in
    the algorithm and must surface where it happened. That is also what the inherited
    collect did, so nothing here changes when a run fails.
    """
    out: dict[str, Any] = {}
    for spec in _OUTPUT_REGISTRY.values():
        out.update(spec.func(algorithm))
    return out


def collect_outputs(algorithm) -> dict:
    """Everything an algorithm emits besides the embedding: registry + its own.

    Algorithm-specific keys win a collision — a method that names a key has made a
    decision about it; a generic extractor has not.
    """
    out = collect_registered_outputs(algorithm)
    fn = getattr(algorithm, "extra_outputs", None)
    if callable(fn) and not getattr(fn, "_manylatents_registry_default", False):
        out.update(fn())
    return out
