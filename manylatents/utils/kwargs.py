"""Route caller kwargs to a downstream estimator: forward the parseable ones, reject the rest.

`LatentModule.__init__` already raises on unexpected keyword arguments (no more silent drops). That
guard is deliberately blunt -- it rejects *every* leftover, including parameters the downstream
estimator would happily accept (e.g. ``spread`` / ``set_op_mix_ratio`` for umap-learn). Rather than
hand-declare every downstream parameter on every module, a module can route its extras through
``route_kwargs``: keys the target's signature accepts are forwarded, keys it does not are raised with a
did-you-mean hint. This keeps the fail-loud contract while letting valid downstream parameters through.
"""
from __future__ import annotations

import difflib
import inspect
import warnings
from typing import Any, Callable, Iterable


def accepted_params(target: Callable) -> set[str]:
    """Explicit keyword parameters ``target`` accepts.

    Deliberately ignores ``target``'s own ``**kwargs``: honoring a downstream ``VAR_KEYWORD`` would
    re-open the silent-swallow hole for estimators that themselves take ``**kwargs``. A module whose
    backend genuinely needs pass-through declares those keys via ``allow=``.
    """
    params = inspect.signature(target).parameters
    return {
        name
        for name, p in params.items()
        if name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def route_kwargs(
    target: Callable,
    extra: dict[str, Any],
    *,
    allow: Iterable[str] = (),
    strict: bool = True,
    context: str = "",
) -> dict[str, Any]:
    """Return the subset of ``extra`` that ``target`` accepts; handle the rest.

    Args:
        target: the downstream callable (e.g. ``umap.UMAP``) whose signature defines valid keys.
        extra: leftover caller kwargs to route.
        allow: extra keys to treat as valid even if not in ``target``'s signature (backend
            pass-through params, framework keys).
        strict: if True (default) raise ``TypeError`` on any unrecognized key; if False, ``warn``.
        context: label for the message (e.g. ``"UMAPModule[umap-learn]"``).

    Raises:
        TypeError: if ``strict`` and any key is neither accepted by ``target`` nor in ``allow``.
    """
    if not extra:
        return {}
    accepted = accepted_params(target) | set(allow)
    forwarded = {k: v for k, v in extra.items() if k in accepted}
    unknown = [k for k in extra if k not in accepted]
    if unknown:
        hints = []
        for k in unknown:
            match = difflib.get_close_matches(k, accepted, n=1)
            hints.append(f"{k!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
        where = f"{context}: " if context else ""
        msg = (
            f"{where}unexpected parameter(s) {', '.join(hints)}. A silently ignored parameter makes "
            "two different configurations return identical results."
        )
        if strict:
            raise TypeError(msg)
        warnings.warn(msg, stacklevel=2)
    return forwarded
