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
from .trajectory_aligner import TrajectoryAligner

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
    "TrajectoryAligner",
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
    """Discover algorithm classes from the .py files in this directory.

    This used to iterate the hand-maintained ``__all__`` above, which made the registry a
    record of what someone remembered to export rather than of what exists. Two working
    modules were invisible because of it — ``aa.py`` (ArchetypalAnalysisModule) and
    ``dr_noop.py`` (NoOpModule) — and a prior audit concluded from ``list_algorithms()`` that
    archetypal analysis did not exist in this engine at all. Two entries were visible that are
    not algorithms: ``ChannelLoadings`` (a dataclass) and ``FoundationEncoder`` (abstract).

    Scanning the filesystem and filtering on ``issubclass(LatentModule)`` fixes both
    directions at once, and is the pattern ``manylatents/data/__init__.py`` already uses.
    ``__all__`` remains the re-export surface; it is simply no longer the source of truth.
    """
    import importlib
    import inspect
    from pathlib import Path

    from .latent_module_base import LatentModule

    global _ALGORITHM_REGISTRY
    _ALGORITHM_REGISTRY.clear()

    here = Path(__file__).parent
    discovered: dict[str, type] = {}
    # Non-recursive glob, so nested test/helper directories are not swept in.
    for path in sorted(here.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        try:
            module = importlib.import_module(f".{path.stem}", package=__name__)
        except Exception:  # noqa: BLE001 - an unimportable module is not an algorithm
            continue
        members = [
            obj for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, LatentModule)
            and obj is not LatentModule
            and not inspect.isabstract(obj)          # drops FoundationEncoder
            and obj.__module__ == module.__name__    # only classes DEFINED here, not imported
        ]
        for obj in members:
            discovered[obj.__name__] = obj
        # The file stem is an alias when a module defines exactly one algorithm — this is what
        # makes the packaged config basenames (`aa`, `noop`) resolve by name.
        if len(members) == 1:
            _ALGORITHM_REGISTRY.setdefault(path.stem.lower(), members[0])

    for obj in discovered.values():
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

    # Canonical names come from the REGISTRY, not from `__all__` — otherwise this reports what
    # someone remembered to export rather than what is actually resolvable, which is exactly
    # how `aa` and `noop` came to be missing from the catalogue while working fine.
    canonical = set()
    for obj in set(_ALGORITHM_REGISTRY.values()):
        class_name = obj.__name__
        base_name = class_name[: -len("Module")] if class_name.endswith("Module") else class_name
        canonical.add(_to_snake_case(base_name))

    return sorted(canonical)