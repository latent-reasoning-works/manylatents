# LatentModule algorithms (fit/transform pattern)
from .pca import PCAModule
from .tsne import TSNEModule
from .umap import UMAPModule
from .phate import PHATEModule
from .multiscale_phate import MultiscalePHATEModule
from .diffusion_map import DiffusionMapModule
from .multi_dimensional_scaling import MDSModule
from .merging import MergingModule, ChannelLoadings
from .classifier import ClassifierModule
from .leiden import LeidenModule
from .reeb_graph import ReebGraphModule
from .selective_correction import SelectiveCorrectionModule
from .foundation_encoder import FoundationEncoder

__all__ = [
    "PCAModule",
    "TSNEModule",
    "UMAPModule",
    "PHATEModule",
    "MultiscalePHATEModule",
    "DiffusionMapModule",
    "MDSModule",
    "MergingModule",
    "ChannelLoadings",
    "ClassifierModule",
    "LeidenModule",
    "ReebGraphModule",
    "SelectiveCorrectionModule",
    "FoundationEncoder",
    "get_algorithm",
    "list_algorithms",
]

import re
from typing import Dict, List, Type

# Registry populated lazily on first access
_ALGORITHM_REGISTRY: Dict[str, Type] = {}


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _build_registry() -> None:
    """Scan __all__ exports and register each class under multiple name variants."""
    global _ALGORITHM_REGISTRY
    _ALGORITHM_REGISTRY.clear()

    current_module_globals = globals()
    for export_name in __all__:
        obj = current_module_globals.get(export_name)
        if obj is None or not isinstance(obj, type):
            continue

        class_name = obj.__name__
        # Base name: strip trailing "Module" if present
        base_name = class_name[: -len("Module")] if class_name.endswith("Module") else class_name
        snake = _to_snake_case(base_name)
        collapsed = base_name.lower()

        variants = {
            class_name.lower(),   # e.g. pcamodule
            base_name.lower(),    # e.g. pca
            snake,                # e.g. diffusion_map
            collapsed,            # e.g. diffusionmap
        }

        for variant in variants:
            if variant not in _ALGORITHM_REGISTRY:
                _ALGORITHM_REGISTRY[variant] = obj


def get_algorithm(name: str) -> Type:
    """Return an algorithm class by name (case-insensitive, supports snake_case).

    Args:
        name: Algorithm name — class name, base name, snake_case, or collapsed
              lowercase all work. Lookup is case-insensitive.

    Returns:
        The algorithm class (not an instance).

    Raises:
        KeyError: If no algorithm matches *name*.
    """
    if not _ALGORITHM_REGISTRY:
        _build_registry()

    key = name.lower().replace("-", "_")

    if key in _ALGORITHM_REGISTRY:
        return _ALGORITHM_REGISTRY[key]

    available = list_algorithms()
    raise KeyError(
        f"Unknown algorithm: '{name}'. Available (snake_case): {available}"
    )


def list_algorithms() -> List[str]:
    """Return a sorted, deduplicated list of canonical algorithm names (snake_case base)."""
    if not _ALGORITHM_REGISTRY:
        _build_registry()

    # Canonical names: snake_case of the base name (no "Module" suffix)
    canonical = set()
    for export_name in __all__:
        obj = globals().get(export_name)
        if obj is None or not isinstance(obj, type):
            continue
        class_name = obj.__name__
        base_name = class_name[: -len("Module")] if class_name.endswith("Module") else class_name
        canonical.add(_to_snake_case(base_name))

    return sorted(canonical)