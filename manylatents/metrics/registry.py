"""Metric registry with decorator-based auto-discovery.

Provides semantic aliases (e.g., "beta_0" -> persistent_homology with homology_dim=0)
and a unified interface for metric lookup and instantiation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


@dataclass
class MetricSpec:
    """Specification for a metric alias."""

    func: Callable  # The actual metric function
    params: Dict[str, Any] = field(default_factory=dict)  # Default params for this alias
    description: str = ""
    #: Which key of a dict return is THE scalar. Without it `_to_scalar` took whichever key
    #: came first, so `connected_components` recorded 66.667 — the mean of the component
    #: SIZES — under a description promising "Number of connected components".
    scalar_key: Optional[str] = None

    def __call__(
        self,
        embeddings: np.ndarray,
        dataset: Optional[object] = None,
        module: Optional[LatentModule] = None,
        cache: Optional[dict] = None,
        **kwargs,
    ) -> Union[float, tuple[float, np.ndarray], Dict[str, Any]]:
        """Call the metric with alias defaults merged with kwargs."""
        merged = {**self.params, **kwargs}
        return self.func(embeddings=embeddings, dataset=dataset, module=module, cache=cache, **merged)


# Global registry
_REGISTRY: Dict[str, MetricSpec] = {}


def register_metric(
    aliases: Optional[List[str]] = None,
    default_params: Optional[Dict[str, Any]] = None,
    description: str = "",
    scalar_key: Optional[str] = None,
):
    """Decorator to register a metric function with optional aliases.

    The function itself is always registered under its own name.
    Additional aliases map to the same function with preset parameters.

    Args:
        aliases: Additional names that map to this metric with default_params.
        default_params: Parameters to apply when using the aliases.
        description: Human-readable description of what this alias computes.
        scalar_key: For a metric returning a dict, which key IS the recorded number. Omit
            only when the dict has exactly one numeric value; otherwise `_to_scalar` raises
            rather than picking by insertion order.

    Example:
        @register_metric(
            aliases=["beta_0", "connected_components"],
            default_params={"homology_dim": 0},
            description="Count of connected components (H0)",
        )
        def PersistentHomology(embeddings, dataset=None, module=None, homology_dim=1, ...):
            ...

    After decoration:
        - "PersistentHomology" -> calls with no preset params
        - "beta_0" -> calls with homology_dim=0
        - "connected_components" -> calls with homology_dim=0
    """

    def decorator(fn: Callable) -> Callable:
        # Always register the function under its own name (no preset params)
        _REGISTRY[fn.__name__] = MetricSpec(func=fn, params={}, description=fn.__doc__ or "",
                                            scalar_key=scalar_key)

        # Register aliases with preset params
        if aliases:
            for alias in aliases:
                _REGISTRY[alias] = MetricSpec(
                    func=fn,
                    params=default_params or {},
                    description=description or fn.__doc__ or "",
                    scalar_key=scalar_key,
                )
                logger.debug(f"Registered metric alias: {alias} -> {fn.__name__}")

        return fn

    return decorator


def get_metric(name: str) -> MetricSpec:
    """Get a metric spec by name or alias.

    Args:
        name: Metric name or alias (e.g., "beta_0", "ParticipationRatio").

    Returns:
        MetricSpec that can be called directly.

    Raises:
        KeyError: If metric not found.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())[:10])
        raise KeyError(f"Metric '{name}' not found. Available: {available}...")
    return _REGISTRY[name]


def resolve_metric(name: str) -> tuple[Callable, Dict[str, Any]]:
    """Resolve a metric alias to (function, default_params).

    Args:
        name: Metric name or alias.

    Returns:
        Tuple of (metric_function, params_dict).
    """
    if name in _REGISTRY:
        spec = _REGISTRY[name]
        return spec.func, spec.params
    return None, {}


def get_metric_registry() -> Dict[str, MetricSpec]:
    """Get a copy of the full metric registry."""
    return _REGISTRY.copy()


def list_metrics() -> List[str]:
    """List all available metric names and aliases."""
    return sorted(_REGISTRY.keys())


def _to_scalar(raw: Any, scalar_key: Optional[str] = None) -> float:
    """Normalize any metric return type to a single float.

    For a dict return, `scalar_key` names which entry IS the metric. Without it this took
    whichever key happened to come first, which is insertion order, which is whatever the
    author wrote first — not a decision anyone made. Three metrics were recording the wrong
    number under a description promising something else: `connected_components` reported
    66.667 (the mean component SIZE) for a 3-component graph, `shepard_residual` reported the
    OLS slope rather than a residual, and `topology_descriptor` reported `n_samples`.

    Falling back is safe only when there is exactly one numeric value and therefore nothing to
    choose between. Otherwise raise: a silently arbitrary pick is how those three survived.
    """
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    if isinstance(raw, tuple) and len(raw) >= 1:
        return float(raw[0])
    if isinstance(raw, np.ndarray):
        return float(np.mean(raw))
    if isinstance(raw, dict):
        if scalar_key is not None:
            if scalar_key not in raw:
                raise ValueError(
                    f"scalar_key {scalar_key!r} not in metric result {sorted(raw)}"
                )
            return float(raw[scalar_key])
        numeric = {k: v for k, v in raw.items()
                   if isinstance(v, (int, float, np.integer, np.floating))}
        if len(numeric) == 1:
            return float(next(iter(numeric.values())))
        if not numeric:
            raise ValueError(f"Dict metric has no scalar value: {list(raw.keys())}")
        raise ValueError(
            f"Ambiguous dict metric with keys {sorted(numeric)} — declare scalar_key on its "
            "register_metric() so the recorded number is chosen rather than stumbled into."
        )
    return float(raw)


def compute_metric(
    name: str,
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute a metric by name, always returning a scalar float.

    Args:
        name: Metric name or alias.
        embeddings: The embedding array.
        dataset: Optional dataset object.
        module: Optional LatentModule.
        cache: Optional shared cache dict for deduplicated computation.
        **kwargs: Additional parameters (override alias defaults).

    Returns:
        Scalar float metric value. For per-sample or structured output,
        use compute_metric_detailed().
    """
    spec = get_metric(name)
    raw = spec(embeddings=embeddings, dataset=dataset, module=module, cache=cache, **kwargs)
    return _to_scalar(raw, spec.scalar_key)


def compute_metric_detailed(
    name: str,
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Compute a metric with full detail (scalar + per-sample + raw).

    Args:
        name: Metric name or alias.
        embeddings: The embedding array.
        dataset: Optional dataset object.
        module: Optional LatentModule.
        cache: Optional shared cache dict for deduplicated computation.
        **kwargs: Additional parameters (override alias defaults).

    Returns:
        Dict with keys:
            - "value": float (scalar summary)
            - "per_sample": np.ndarray or None
            - "raw": original return value from the metric function
    """
    spec = get_metric(name)
    raw = spec(embeddings=embeddings, dataset=dataset, module=module, cache=cache, **kwargs)

    per_sample = None
    if isinstance(raw, tuple) and len(raw) >= 2:
        per_sample = np.asarray(raw[1]) if raw[1] is not None else None
    elif isinstance(raw, np.ndarray) and raw.ndim >= 1:
        per_sample = raw

    return {
        "value": _to_scalar(raw, spec.scalar_key),
        "per_sample": per_sample,
        "raw": raw,
    }
