"""Parameter declarations: which values each parameter of a method accepts, as plain data.

The format (declaration version 1) and its two checks, :func:`check_declaration` and
:func:`check`, belong to manykinds: see ``manykinds.params`` and
https://github.com/latent-reasoning-works/manykinds/blob/main/docs/param-declarations.md.
They are re-exported here unchanged, so a caller of manylatents needs one import. A
declaration file is checked with ``python -m manykinds.params check FILE``.

manylatents produces declarations for its own methods: ``legal.type`` and ``default`` come from
the constructor signature (latent modules) or from the composed config (Lightning configs). The
rest comes from a source: the class attribute ``PARAMS`` of a latent module, or the file
``configs/algorithms/lightning/domains/<config>.yaml``. A source entry never holds ``default``.

Reading a Lightning declaration composes the config with Hydra; it imports no Lightning module.
"""
from __future__ import annotations

import copy
import inspect
import types
import typing
from typing import Any

from manykinds.params import SCHEMA_VERSION, TYPES, check, check_declaration

__all__ = ["SCHEMA_VERSION", "check", "check_declaration", "domain", "names"]

# Keys a source entry (PARAMS or a Lightning domain file) may hold. ``type`` is needed only when
# the signature or the config value does not give it. A source entry never holds ``default``.
SOURCE_KEYS = (
    "type", "min", "min_open", "max", "max_open", "choices", "also",
    "sensible", "ladder", "limits", "role", "precomputed", "cost", "meaning", "more", "less",
)
_SOURCE_DEFAULTS: dict[str, Any] = {
    "type": None, "min": None, "min_open": False, "max": None, "max_open": False,
    "choices": [], "also": [], "sensible": None, "ladder": None, "limits": [],
    "precomputed": False, "cost": None, "more": None, "less": None,
}

ORIGIN = "manylatents"
_DOCS = "docs/algorithms.md, section 'Parameter declarations'"


def _listed(items: Any, value: Any) -> bool:
    """True when ``value`` is in ``items`` with the same type (so True is not 1)."""
    return isinstance(items, list) and any(
        type(item) is type(value) and item == value for item in items
    )


# ---------------------------------------------------------------------------
# Building an entry from a source and a signature or config value
# ---------------------------------------------------------------------------

_PY_TYPES = {int: "int", float: "float", bool: "bool", str: "str"}


def _annotation_types(annotation: Any) -> set[str] | None:
    """The simple types an annotation allows, without None. None when nothing is known."""
    if annotation is inspect.Parameter.empty or isinstance(annotation, str):
        return None
    if annotation is None or annotation is type(None):
        return set()
    if annotation in _PY_TYPES:
        return {_PY_TYPES[annotation]}
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        found: set[str] = set()
        for arg in typing.get_args(annotation):
            sub = _annotation_types(arg)
            if sub is None:
                continue
            found |= sub
        return found
    return set()  # a container, a class or an array: not a simple value


def _value_type(value: Any) -> set[str] | None:
    if value is None:
        return None
    for py, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if isinstance(value, py):
            return {name}
    return set()


def _resolve_type(where: str, declared: str | None, allowed: set[str] | None) -> str:
    """Pick ``legal.type`` from the source and from the signature or the config value."""
    if declared is not None:
        if declared not in TYPES:
            raise ValueError(f"{where}: type {declared!r} is not one of {', '.join(TYPES)}")
        if allowed:  # the signature or the value says something: they must agree
            compatible = {
                "int": {"int"}, "float": {"float", "int"}, "bool": {"bool"},
                "choice": {"str", "int", "float"}, "str": {"str"},
            }[declared]
            if not allowed & compatible:
                raise ValueError(
                    f"{where}: the source gives type {declared!r} but the signature or the "
                    f"config value gives {', '.join(sorted(allowed))}"
                )
        return declared
    if not allowed:
        raise ValueError(
            f"{where}: the signature or the config value gives no simple type; "
            "the source must give 'type'"
        )
    if len(allowed) == 1:
        return next(iter(allowed))
    numeric = allowed - {"str"}
    if len(numeric) == 1 and "str" in allowed:
        return next(iter(numeric))  # e.g. int | str with also: ["auto"]
    if numeric == {"int", "float"} and "str" not in allowed:
        return "float"
    raise ValueError(f"{where}: types {sorted(allowed)} are ambiguous; the source must give 'type'")


def _entry(where: str, source: Any, allowed: set[str] | None, default: Any) -> dict:
    if not isinstance(source, dict):
        raise ValueError(f"{where}: the source entry must be a mapping")  # noqa: TRY004
    if "default" in source:
        raise ValueError(
            f"{where}: the source repeats the default; it comes from the signature or the config"
        )
    unknown = sorted(set(source) - set(SOURCE_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown source keys {unknown}")
    for key in ("role", "meaning"):
        if key not in source:
            raise ValueError(f"{where}: the source must give {key!r}")
    merged = {**copy.deepcopy(_SOURCE_DEFAULTS), **copy.deepcopy(source)}
    kind = _resolve_type(where, merged["type"], allowed)
    legal = {
        "type": kind,
        "min": merged["min"], "min_open": merged["min_open"],
        "max": merged["max"], "max_open": merged["max_open"],
        "choices": list(merged["choices"]), "also": list(merged["also"]),
    }
    return {
        "legal": legal,
        "default": default,
        "sensible": merged["sensible"],
        "ladder": merged["ladder"],
        "limits": list(merged["limits"]),
        "role": merged["role"],
        "precomputed": merged["precomputed"],
        "cost": merged["cost"],
        "meaning": merged["meaning"],
        "more": merged["more"],
        "less": merged["less"],
    }


def _declaration(method: str, group: str, entries: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "group": group,
        "origin": ORIGIN,
        "params": entries,
    }


# ---------------------------------------------------------------------------
# Latent modules
# ---------------------------------------------------------------------------

def _canonical_latent_name(cls: type) -> str:
    from manylatents.algorithms.latent import _to_snake_case

    base = cls.__name__.removesuffix("Module")
    return _to_snake_case(base)


def _signature(cls: type) -> inspect.Signature:
    """The constructor signature, with string annotations evaluated where possible."""
    try:
        return inspect.signature(cls.__init__, eval_str=True)
    except Exception:  # noqa: BLE001 - an annotation that does not evaluate gives no type
        return inspect.signature(cls.__init__)


def _latent_declaration(cls: type, method: str) -> dict:
    """Build the declaration of latent module ``cls`` from its signature and ``PARAMS``."""
    source = getattr(cls, "PARAMS", None)
    if source is None:
        raise ValueError(
            f"{cls.__module__}.{cls.__qualname__} declares no PARAMS, so it has no parameter "
            f"declaration. Add a PARAMS class attribute ({_DOCS})."
        )
    sig = _signature(cls)
    wanted = [
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]
    missing = [name for name in wanted if name not in source]
    extra = [name for name in source if name not in wanted]
    if missing or extra:
        raise ValueError(
            f"{cls.__qualname__}.PARAMS must have one entry per constructor parameter: "
            f"missing {missing}, not in the constructor {extra}"
        )
    entries = {}
    for name in wanted:
        param = sig.parameters[name]
        default = None if param.default is inspect.Parameter.empty else param.default
        entries[name] = _entry(
            f"{cls.__qualname__}.PARAMS[{name!r}]", source[name],
            _annotation_types(param.annotation), default,
        )
    return _declaration(method, "latent", entries)


def _latent_class_from_config(name: str) -> type:
    """Resolve ``algorithms/latent/<name>.yaml`` on the Hydra search path to its class."""
    from hydra.errors import HydraException
    from hydra.utils import get_class

    try:
        cfg = _compose([f"algorithms/latent={name}"], extensions=True)
    except HydraException as exc:
        raise KeyError(f"no latent method or config named {name!r}") from exc
    node = cfg.get("algorithms", {}).get("latent")
    target = node.get("_target_") if node is not None else None
    if not target:
        raise KeyError(f"config algorithms/latent/{name}.yaml has no _target_")
    return get_class(target)


# ---------------------------------------------------------------------------
# Lightning configs
# ---------------------------------------------------------------------------

def _configs_dir():
    from importlib.resources import files

    return files("manylatents.configs")


def _register_extensions() -> None:
    """Register the search paths of installed extensions (entry point group manylatents.extensions)."""
    from importlib.metadata import entry_points

    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    plugins = Plugins.instance()
    existing = set(plugins.discover(SearchPathPlugin))
    for ep in entry_points(group="manylatents.extensions"):
        try:
            cls = ep.load()
        except Exception:  # noqa: BLE001, S112 - a broken extension must not break a declaration
            continue
        if isinstance(cls, type) and issubclass(cls, SearchPathPlugin) and cls not in existing:
            plugins.register(cls)


def _compose(overrides: list[str], extensions: bool = False):
    """Compose the root config with ``overrides``, leaving any global Hydra state as it was."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import manylatents.configs  # noqa: F401 - registers base_config and the search path

    if extensions:
        _register_extensions()
    config_dir = str(Path(str(_configs_dir())).resolve())
    gh = GlobalHydra.instance()
    previous = gh.hydra if gh.is_initialized() else None
    gh.clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose(config_name="config", overrides=overrides)
    finally:
        gh = GlobalHydra.instance()
        gh.clear()
        if previous is not None:
            gh.initialize(previous)


def _lightning_config_names() -> list[str]:
    folder = _configs_dir() / "algorithms" / "lightning"
    return sorted(
        entry.name[: -len(".yaml")] for entry in folder.iterdir()
        if entry.is_file() and entry.name.endswith(".yaml")
        and entry.name != "default.yaml" and not entry.name.startswith("_")
    )


def _compose_lightning(name: str) -> dict:
    """The composed Lightning config as plain data: the module node plus ``trainer``.

    Keys under the module node (``network``, ``optimizer``, ``loss``, the module's own
    arguments) are relative to ``algorithms.lightning``; ``trainer`` is the trainer group
    ``trainer=default`` that every Lightning run composes.
    """
    from omegaconf import OmegaConf

    cfg = _compose([f"algorithms/lightning={name}", "trainer=default"])
    module = OmegaConf.to_container(cfg.algorithms.lightning, resolve=False)
    trainer = OmegaConf.to_container(cfg.trainer, resolve=False)
    composed = dict(module)
    if "trainer" in composed:
        raise ValueError(f"lightning config {name!r} has its own 'trainer' key")
    composed["trainer"] = trainer
    return composed


_NOT_DECLARED = frozenset({"datamodule", "input_dim"})


def _lightning_scalar_paths(composed: dict, prefix: str = "") -> list[str]:
    """Every dotted path to a scalar value that a caller may set.

    Skipped: keys that start with ``_`` (``_target_``, ``_partial_``, ``_recursive_``),
    ``datamodule`` and ``input_dim`` (set from the data), interpolations, lists and the
    trainer's ``logger`` and ``callbacks`` subtrees.
    """
    paths = []
    for key, value in composed.items():
        if key.startswith("_") or key in _NOT_DECLARED:
            continue
        if prefix == "trainer." and key in ("logger", "callbacks"):
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            paths.extend(_lightning_scalar_paths(value, path + "."))
        elif isinstance(value, list) or isinstance(value, str) and "${" in value:
            continue
        else:
            paths.append(path)
    return paths


_MISSING = object()


def _lookup(composed: dict, path: str) -> Any:
    node: Any = composed
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _lightning_source(name: str) -> dict:
    import yaml

    path = _configs_dir() / "algorithms" / "lightning" / "domains" / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(
            f"lightning config {name!r} has no domain file "
            f"configs/algorithms/lightning/domains/{name}.yaml ({_DOCS})"
        )
    source = yaml.safe_load(path.read_text()) or {}
    if not isinstance(source, dict):
        raise ValueError(f"domains/{name}.yaml must be a mapping of override paths")  # noqa: TRY004
    return source


def _lightning_declaration(name: str) -> dict:
    if name not in _lightning_config_names():
        raise KeyError(f"no lightning config named {name!r}")
    composed = _compose_lightning(name)
    source = _lightning_source(name)
    entries = {}
    for path, entry in source.items():
        value = _lookup(composed, path)
        where = f"domains/{name}.yaml[{path!r}]"
        if value is _MISSING:
            raise ValueError(f"{where}: the path does not resolve in the composed config")
        if isinstance(value, (dict, list)) or (isinstance(value, str) and "${" in value):
            raise ValueError(f"{where}: the composed value is not a simple value")
        # A value listed in ``also`` (for example "auto" for an int) says nothing about the type.
        in_also = isinstance(entry, dict) and _listed(entry.get("also"), value)
        entries[path] = _entry(where, entry, None if in_also else _value_type(value), value)
    return _declaration(name, "lightning", entries)


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def domain(name: str, group: str = "latent") -> dict:
    """Return the declaration of one method, as plain data.

    A latent name resolves through the registry (``get_algorithm``), then through a config
    ``algorithms/latent/<name>.yaml`` on the Hydra search path, extensions included. A Lightning
    name is the stem of a config in ``configs/algorithms/lightning/``.

    Raises ``KeyError`` for an unknown name and ``ValueError`` for a method without a source.
    """
    if group == "latent":
        from manylatents.algorithms.latent import get_algorithm

        try:
            cls = get_algorithm(name)
            method = _canonical_latent_name(cls)
        except KeyError:
            cls = _latent_class_from_config(name)
            method = name
        return _latent_declaration(cls, method)
    if group == "lightning":
        return _lightning_declaration(name)
    raise ValueError(f"group must be 'latent' or 'lightning', got {group!r}")


def names(group: str = "latent") -> list[str]:
    """Return every method name of ``group`` that has a declaration, sorted."""
    if group == "latent":
        from manylatents.algorithms.latent import get_algorithm, list_algorithms

        return sorted(n for n in list_algorithms() if "PARAMS" in vars(get_algorithm(n)))
    if group == "lightning":
        folder = _configs_dir() / "algorithms" / "lightning" / "domains"
        return [n for n in _lightning_config_names() if (folder / f"{n}.yaml").is_file()]
    raise ValueError(f"group must be 'latent' or 'lightning', got {group!r}")
