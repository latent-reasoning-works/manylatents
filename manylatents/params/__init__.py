"""Parameter declarations: which values each parameter of a method accepts, as plain data.

A declaration (version 1) is a mapping::

    {"schema_version": 1, "method": <name>, "group": "latent" | "lightning" | "tool",
     "origin": <producer>, "params": {<name>: <parameter entry>}}

and a parameter entry has exactly the keys ``legal``, ``default``, ``sensible``, ``ladder``,
``limits``, ``role``, ``precomputed``, ``cost``, ``meaning``, ``more`` and ``less``. The
format is defined by the prose in docs/algorithms.md ("Parameter declarations"), by
:func:`check_declaration` and by the fixtures in ``tests/fixtures/param_declarations/``.

manylatents produces declarations for its own methods: ``legal.type`` and ``default`` come from
the constructor signature (latent modules) or from the composed config (Lightning configs). The
rest comes from a source: the class attribute ``PARAMS`` of a latent module, or the file
``configs/algorithms/lightning/domains/<config>.yaml``. A source entry never holds ``default``.

Reading a Lightning declaration composes the config with Hydra; it imports no Lightning module.
"""
from __future__ import annotations

import copy
import inspect
import math
import numbers
import types
import typing
from typing import Any

__all__ = ["check", "check_declaration", "domain", "names"]

SCHEMA_VERSION = 1
GROUPS = ("latent", "lightning", "tool")
TYPES = ("int", "float", "bool", "choice", "str")
ROLES = ("knob", "fixed", "infra")
SCALES = ("linear", "log")
SOURCE_KINDS = ("docs", "paper", "measured")
FACTS = ("n_samples", "n_features")
OPS = ("le", "lt")
COST_MOVES = ("up", "down", "none")
TEXT_MAX = 120

DOC_KEYS = ("schema_version", "method", "group", "origin", "params")
ENTRY_KEYS = (
    "legal", "default", "sensible", "ladder", "limits", "role", "precomputed", "cost",
    "meaning", "more", "less",
)
LEGAL_KEYS = ("type", "min", "min_open", "max", "max_open", "choices", "also")
SENSIBLE_KEYS = ("lo", "hi", "scale", "source")
SOURCE_REF_KEYS = ("kind", "ref")
LIMIT_KEYS = ("op", "fact", "offset")
COST_KEYS = ("time", "memory", "note")

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


# ---------------------------------------------------------------------------
# Values against legal
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _same(a: Any, b: Any) -> bool:
    """Equality that keeps bools, numbers, strings and None apart."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is None and b is None
    if _is_number(a) and _is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def _contains(items: Any, value: Any) -> bool:
    return isinstance(items, list) and any(_same(value, item) for item in items)


def _legal_refusal(legal: dict, value: Any) -> str | None:
    """Return why ``legal`` refuses ``value``, or None when it accepts it."""
    if _contains(legal.get("also"), value):
        return None
    kind = legal.get("type")
    choices = legal.get("choices") or []
    if kind == "bool":
        return None if isinstance(value, bool) else "not a bool"
    if kind == "choice":
        return None if _contains(choices, value) else "not one of the choices"
    if kind == "str":
        if not isinstance(value, str):
            return "not a str"
        if choices and not _contains(choices, value):
            return "not one of the choices"
        return None
    if kind == "int":
        if not (isinstance(value, numbers.Integral) and not isinstance(value, bool)):
            return "not an int"
    elif kind == "float":
        if not _is_number(value):
            return "not a number"
        if isinstance(value, float) and math.isnan(value):
            return "not a number"
    else:
        return "legal.type is not known"
    lo, hi = legal.get("min"), legal.get("max")
    if lo is not None and (value < lo or (legal.get("min_open") and value == lo)):
        return f"below min {lo}" if value < lo else f"not above min {lo}"
    if hi is not None and (value > hi or (legal.get("max_open") and value == hi)):
        return f"above max {hi}" if value > hi else f"not below max {hi}"
    return None


def _limit_refusal(limit: dict, value: Any, facts: dict[str, int | None]) -> str | None:
    fact = limit.get("fact")
    known = facts.get(fact)
    if known is None or not _is_number(value):
        return None
    bound = known + limit.get("offset", 0)
    op = limit.get("op")
    ok = value <= bound if op == "le" else value < bound
    if ok:
        return None
    sign = "<=" if op == "le" else "<"
    offset = limit.get("offset", 0)
    rhs = fact if not offset else f"{fact} {'+' if offset > 0 else '-'} {abs(offset)}"
    return f"must be {sign} {rhs} ({fact}={known})"


# ---------------------------------------------------------------------------
# check_declaration
# ---------------------------------------------------------------------------

def _keys(where: str, mapping: dict, allowed: tuple[str, ...], problems: list[str]) -> bool:
    ok = True
    for key in mapping:
        if key not in allowed:
            problems.append(f"{_path(where, key)}: unknown key {key!r}")
            ok = False
    for key in allowed:
        if key not in mapping:
            problems.append(f"{_path(where, key)}: missing key {key!r}")
            ok = False
    return ok


def _path(where: str, key: Any) -> str:
    return f"{where}.{key}" if where else str(key)


def _one_of(where: str, value: Any, allowed: tuple[str, ...], problems: list[str]) -> bool:
    if isinstance(value, str) and value in allowed:
        return True
    problems.append(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    return False


def _text(where: str, value: Any, nullable: bool, problems: list[str]) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not 1 <= len(value) <= TEXT_MAX:
        prefix = "null or " if nullable else ""
        problems.append(f"{where}: must be {prefix}1 to {TEXT_MAX} characters")


def _check_legal(where: str, legal: Any, problems: list[str]) -> bool:
    """Check the shape of ``legal``. Return True when values can be tested against it."""
    if not isinstance(legal, dict):
        problems.append(f"{where}: expected a mapping")
        return False
    if not _keys(where, legal, LEGAL_KEYS, problems):
        return False
    ok = _one_of(f"{where}.type", legal["type"], TYPES, problems)
    for key in ("min", "max"):
        if legal[key] is not None and not _is_number(legal[key]):
            problems.append(f"{where}.{key}: expected a number or null, got {legal[key]!r}")
            ok = False
    for key in ("min_open", "max_open"):
        if not isinstance(legal[key], bool):
            problems.append(f"{where}.{key}: expected a bool, got {legal[key]!r}")
            ok = False
    for key in ("choices", "also"):
        if not isinstance(legal[key], list):
            problems.append(f"{where}.{key}: expected a list, got {legal[key]!r}")
            ok = False
    if not ok:
        return False
    if legal["type"] not in ("int", "float") and (legal["min"] is not None or legal["max"] is not None):
        problems.append(f"{where}: min and max are only for int and float")
    if legal["type"] not in ("choice", "str") and legal["choices"]:
        problems.append(f"{where}.choices: only for choice and str")
    if legal["type"] == "choice" and not legal["choices"] and not legal["also"]:
        problems.append(f"{where}.choices: a choice needs at least one choice")
    if (legal["min"] is not None and legal["max"] is not None and legal["min"] > legal["max"]):
        problems.append(f"{where}: min {legal['min']} is greater than max {legal['max']}")
    return True


def _check_value(where: str, legal: dict, value: Any, problems: list[str]) -> None:
    why = _legal_refusal(legal, value)
    if why is not None:
        problems.append(f"{where}: {value!r} is refused by legal ({why})")


def _check_entry(where: str, entry: Any, problems: list[str]) -> None:
    if not isinstance(entry, dict):
        problems.append(f"{where}: expected a mapping")
        return
    _keys(where, entry, ENTRY_KEYS, problems)
    legal_ok = "legal" in entry and _check_legal(f"{where}.legal", entry["legal"], problems)
    legal = entry.get("legal")
    numeric = legal_ok and legal["type"] in ("int", "float")

    if legal_ok and "default" in entry:
        _check_value(f"{where}.default", legal, entry["default"], problems)

    ladder = entry.get("ladder")
    if ladder is not None:
        if not isinstance(ladder, list) or not ladder:
            problems.append(f"{where}.ladder: expected null or a non-empty list")
        elif legal_ok:
            for i, value in enumerate(ladder):
                _check_value(f"{where}.ladder[{i}]", legal, value, problems)

    sensible = entry.get("sensible")
    if sensible is not None:
        _check_sensible(f"{where}.sensible", sensible, legal if legal_ok else None, numeric, problems)

    limits = entry.get("limits", [])
    if not isinstance(limits, list):
        problems.append(f"{where}.limits: expected a list")
    else:
        for i, limit in enumerate(limits):
            lw = f"{where}.limits[{i}]"
            if not isinstance(limit, dict):
                problems.append(f"{lw}: expected a mapping")
                continue
            if not _keys(lw, limit, LIMIT_KEYS, problems):
                continue
            _one_of(f"{lw}.op", limit["op"], OPS, problems)
            _one_of(f"{lw}.fact", limit["fact"], FACTS, problems)
            offset = limit["offset"]
            if not isinstance(offset, numbers.Integral) or isinstance(offset, bool):
                problems.append(f"{lw}.offset: expected an int, got {offset!r}")
            if legal_ok and not numeric:
                problems.append(f"{lw}: limits are only for int and float")

    role_ok = "role" in entry and _one_of(f"{where}.role", entry["role"], ROLES, problems)
    precomputed = entry.get("precomputed", False)
    if not isinstance(precomputed, bool):
        problems.append(f"{where}.precomputed: expected a bool, got {precomputed!r}")
    elif precomputed and role_ok and entry["role"] != "fixed":
        problems.append(
            f"{where}.precomputed: precomputed is true but role is {entry['role']!r}, not 'fixed'"
        )

    cost = entry.get("cost")
    if cost is not None:
        if not isinstance(cost, dict):
            problems.append(f"{where}.cost: expected null or a mapping")
        elif _keys(f"{where}.cost", cost, COST_KEYS, problems):
            _one_of(f"{where}.cost.time", cost["time"], COST_MOVES, problems)
            _one_of(f"{where}.cost.memory", cost["memory"], COST_MOVES, problems)
            _text(f"{where}.cost.note", cost["note"], True, problems)

    if "meaning" in entry:
        _text(f"{where}.meaning", entry["meaning"], False, problems)
    for key in ("more", "less"):
        if key in entry:
            _text(f"{where}.{key}", entry[key], True, problems)


def _check_sensible(where: str, sensible: Any, legal: dict | None, numeric: bool,
                    problems: list[str]) -> None:
    if not isinstance(sensible, dict):
        problems.append(f"{where}: expected null or a mapping")
        return
    if not _keys(where, sensible, SENSIBLE_KEYS, problems):
        return
    if legal is not None and not numeric:
        problems.append(f"{where}: sensible is only for int and float")
        return
    lo, hi, scale = sensible["lo"], sensible["hi"], sensible["scale"]
    bounds_ok = True
    for key, value in (("lo", lo), ("hi", hi)):
        if not _is_number(value):
            problems.append(f"{where}.{key}: expected a number, got {value!r}")
            bounds_ok = False
        elif legal is not None:
            _check_value(f"{where}.{key}", legal, value, problems)
    if bounds_ok and lo > hi:
        problems.append(f"{where}: lo {lo} is greater than hi {hi}")
    if _one_of(f"{where}.scale", scale, SCALES, problems) and scale == "log" and bounds_ok and lo <= 0:
        problems.append(f"{where}: scale log needs lo > 0, got lo {lo}")
    source = sensible["source"]
    if not isinstance(source, dict):
        problems.append(f"{where}.source: expected a mapping")
    elif _keys(f"{where}.source", source, SOURCE_REF_KEYS, problems):
        _one_of(f"{where}.source.kind", source["kind"], SOURCE_KINDS, problems)
        if not isinstance(source["ref"], str) or not source["ref"].strip():
            problems.append(f"{where}.source.ref: expected a non-empty text")


def check_declaration(doc: Any) -> list[str]:
    """Return one message per problem of a declaration, and ``[]`` for a valid one.

    ``doc`` is plain data (for example ``yaml.safe_load`` or ``json.load`` output) from any
    producer. The rules are those of docs/algorithms.md, section "Parameter declarations".
    """
    if not isinstance(doc, dict):
        return [f"declaration: expected a mapping, got {type(doc).__name__}"]
    problems: list[str] = []
    _keys("", doc, DOC_KEYS, problems)
    if "schema_version" in doc and (
        doc["schema_version"] != SCHEMA_VERSION or isinstance(doc["schema_version"], bool)
    ):
        problems.append(f"schema_version: {doc['schema_version']!r} is not {SCHEMA_VERSION}")
    for key in ("method", "origin"):
        if key in doc and (not isinstance(doc[key], str) or not doc[key].strip()):
            problems.append(f"{key}: expected a non-empty text, got {doc[key]!r}")
    if "group" in doc:
        _one_of("group", doc["group"], GROUPS, problems)
    if "params" in doc:
        entries = doc["params"]
        if not isinstance(entries, dict):
            problems.append("params: expected a mapping")
        else:
            for name, entry in entries.items():
                if not isinstance(name, str) or not name:
                    problems.append(f"params: parameter name {name!r} is not a non-empty text")
                    continue
                _check_entry(f"params.{name}", entry, problems)
    return problems


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def check(doc: dict, values: dict, *, n_samples: int | None = None,
          n_features: int | None = None) -> list[str]:
    """Return one message per value in ``values`` that the declaration ``doc`` refuses.

    A value is refused when its parameter is unknown, when its role is not ``knob``, when
    ``legal`` refuses it, or when a limit refuses it and the limit's fact is given. ``sensible``
    and ``ladder`` are never read: any legal value may be submitted.
    """
    entries = doc.get("params", {})
    facts = {"n_samples": n_samples, "n_features": n_features}
    problems: list[str] = []
    for name, value in values.items():
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: not a parameter of {doc.get('method')!r}")
            continue
        role = entry.get("role")
        if role != "knob":
            problems.append(f"{name}: its role is {role}, so it is not offered; only a knob is")
            continue
        why = _legal_refusal(entry["legal"], value)
        if why is not None:
            problems.append(f"{name}={value!r}: refused by legal ({why})")
            continue
        for limit in entry.get("limits", []):
            why = _limit_refusal(limit, value, facts)
            if why is not None:
                problems.append(f"{name}={value!r}: {why}")
    return problems


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
        in_also = isinstance(entry, dict) and _contains(entry.get("also"), value)
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
