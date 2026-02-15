"""mkdocs-macros hook: auto-generate tables from configs and metric registry.

This module is loaded by mkdocs-macros-plugin via the `module_name` setting
in mkdocs.yml. It exposes Jinja2 macros that markdown pages call to inject
auto-generated tables.

Can also be imported directly for testing (functions prefixed with _).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Root of the manylatents package (one level up from docs/)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = _PACKAGE_ROOT / "manylatents" / "configs"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict on error."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def _class_name_from_target(target: str) -> str:
    """Extract class name from a Hydra _target_ string."""
    return target.rsplit(".", 1)[-1] if target else "?"


def _config_name(path: Path) -> str:
    """Extract config override name from a YAML file path."""
    return path.stem


def _skip_file(path: Path) -> bool:
    """Skip non-config files."""
    return path.name.startswith("_") or path.name == "default.yaml" or not path.suffix == ".yaml"


# ---------------------------------------------------------------------------
# Algorithm table
# ---------------------------------------------------------------------------

def _algorithm_table(algo_type: str) -> str:
    """Build markdown table for algorithms of a given type (latent or lightning).

    Walks configs/algorithms/{algo_type}/*.yaml.
    Columns: algorithm | type | config | key params
    """
    config_dir = _CONFIGS / "algorithms" / algo_type
    if not config_dir.is_dir():
        return f"*No configs found at `configs/algorithms/{algo_type}/`*"

    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        if _skip_file(path):
            continue
        cfg = _load_yaml(path)
        target = cfg.get("_target_", "")
        name = _class_name_from_target(target)
        override = f"`algorithms/{algo_type}={_config_name(path)}`"

        # Extract interesting params (skip internal ones)
        skip_keys = {"_target_", "_partial_", "n_components", "random_state",
                      "neighborhood_size", "defaults"}
        params = [k for k in cfg if k not in skip_keys and not k.startswith("_")]
        param_str = ", ".join(f"`{p}`" for p in params[:4]) or "--"

        rows.append(f"| {name} | `{algo_type}` | {override} | {param_str} |")

    if not rows:
        return f"*No algorithm configs found in `configs/algorithms/{algo_type}/`*"

    header = "| algorithm | type | config | key params |\n|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def _get_registry_descriptions() -> dict[str, str]:
    """Get metric descriptions from the registry (if available)."""
    try:
        from manylatents.metrics.registry import get_metric_registry
        registry = get_metric_registry()
        descs = {}
        for alias, spec in registry.items():
            func_name = spec.func.__name__
            desc = spec.description.strip().split("\n")[0] if spec.description else ""
            if desc and (func_name not in descs or len(desc) < len(descs.get(func_name, ""))):
                descs[alias] = desc
                descs[func_name] = desc
        return descs
    except ImportError:
        logger.warning("Could not import metric registry; descriptions unavailable")
        return {}


def _metrics_table(context: str) -> str:
    """Build markdown table for metrics of a given context (embedding/module/dataset).

    Walks configs/metrics/{context}/*.yaml and cross-references the metric registry.
    Columns: metric | config | default params | description
    """
    config_dir = _CONFIGS / "metrics" / context
    if not config_dir.is_dir():
        return f"*No configs found at `configs/metrics/{context}/`*"

    descs = _get_registry_descriptions()
    rows = []

    for path in sorted(config_dir.glob("*.yaml")):
        if _skip_file(path):
            continue

        cfg = _load_yaml(path)
        config_name = _config_name(path)

        # Metric configs are nested: {metric_name: {_target_: ..., ...}}
        inner = None
        metric_key = config_name
        for key, val in cfg.items():
            if isinstance(val, dict) and "_target_" in val:
                inner = val
                metric_key = key
                break

        if inner is None:
            inner = cfg
            if "_target_" not in inner:
                continue

        target = inner.get("_target_", "")
        func_name = _class_name_from_target(target)
        override = f"`metrics/{context}={config_name}`"

        # Extract default params
        skip_keys = {"_target_", "_partial_"}
        params = {k: v for k, v in inner.items() if k not in skip_keys}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "--"

        # Get description from registry
        desc = descs.get(metric_key, descs.get(func_name, ""))
        if len(desc) > 80:
            desc = desc[:77] + "..."

        rows.append(f"| {func_name} | {override} | {param_str} | {desc} |")

    if not rows:
        return f"*No metric configs found in `configs/metrics/{context}/`*"

    header = "| metric | config | defaults | description |\n|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

def _data_table() -> str:
    """Build markdown table for data modules.

    Walks configs/data/*.yaml.
    Columns: dataset | config | key params
    """
    config_dir = _CONFIGS / "data"
    if not config_dir.is_dir():
        return "*No configs found at `configs/data/`*"

    skip_names = {"default", "test_data"}
    rows = []

    for path in sorted(config_dir.glob("*.yaml")):
        if _skip_file(path) or path.stem in skip_names:
            continue

        cfg = _load_yaml(path)
        target = cfg.get("_target_", "")
        if not target:
            continue

        name = _class_name_from_target(target).replace("DataModule", "")
        override = f"`data={_config_name(path)}`"

        skip_keys = {"_target_", "_partial_", "defaults", "random_state", "test_split"}
        params = {k: v for k, v in cfg.items()
                  if k not in skip_keys and not k.startswith("_") and not isinstance(v, dict)}
        param_items = list(params.items())[:3]
        param_str = ", ".join(f"{k}={v}" for k, v in param_items) or "--"

        rows.append(f"| {name} | {override} | {param_str} |")

    if not rows:
        return "*No data configs found in `configs/data/`*"

    header = "| dataset | config | key params |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Sampling table
# ---------------------------------------------------------------------------

def _sampling_table() -> str:
    """Build markdown table for sampling strategies.

    Walks configs/metrics/sampling/*.yaml.
    Columns: strategy | config | key params
    """
    config_dir = _CONFIGS / "metrics" / "sampling"
    if not config_dir.is_dir():
        return "*No configs found at `configs/metrics/sampling/`*"

    rows = []
    for path in sorted(config_dir.glob("*.yaml")):
        if _skip_file(path):
            continue

        cfg = _load_yaml(path)
        inner = cfg.get("sampling", cfg)
        target = inner.get("_target_", "")
        if not target:
            continue

        name = _class_name_from_target(target)
        override = f"`sampling/{_config_name(path)}`"

        skip_keys = {"_target_", "_partial_"}
        params = {k: v for k, v in inner.items() if k not in skip_keys}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "--"

        rows.append(f"| {name} | {override} | {param_str} |")

    if not rows:
        return "*No sampling configs found*"

    header = "| strategy | config | defaults |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# mkdocs-macros entry point
# ---------------------------------------------------------------------------

def define_env(env):
    """mkdocs-macros hook. Registers Jinja2 macros for use in markdown pages."""

    @env.macro
    def algorithm_table(algo_type: str = "latent") -> str:
        return _algorithm_table(algo_type)

    @env.macro
    def metrics_table(context: str = "embedding") -> str:
        return _metrics_table(context)

    @env.macro
    def data_table() -> str:
        return _data_table()

    @env.macro
    def sampling_table() -> str:
        return _sampling_table()
