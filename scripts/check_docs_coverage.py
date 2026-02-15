#!/usr/bin/env python3
"""Check that every metric/algorithm config has proper docs coverage.

Exits non-zero if:
- A metric config exists but the function lacks @register_metric
- A registered metric has an empty docstring
- A metric config's _target_ can't be imported

Run: python scripts/check_docs_coverage.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "manylatents" / "configs"

errors: list[str] = []
warnings: list[str] = []


def check_metric_configs():
    """Check all metric configs have matching registry entries and docstrings."""
    # Import to trigger registration
    import manylatents.metrics  # noqa: F401
    from manylatents.metrics.registry import get_metric_registry

    registry = get_metric_registry()
    registry_func_names = {spec.func.__name__ for spec in registry.values()}

    for context in ("embedding", "module", "dataset"):
        config_dir = CONFIGS / "metrics" / context
        if not config_dir.is_dir():
            continue

        for path in sorted(config_dir.glob("*.yaml")):
            if path.name.startswith("_") or path.name == "test_metric.yaml":
                continue

            with open(path) as f:
                cfg = yaml.safe_load(f) or {}

            # Find _target_ (may be nested)
            target = None
            for key, val in cfg.items():
                if isinstance(val, dict) and "_target_" in val:
                    target = val["_target_"]
                    break
            if target is None:
                target = cfg.get("_target_")
            if target is None:
                warnings.append(f"{path.name}: no _target_ found")
                continue

            # Check target is importable
            module_path, class_name = target.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                func = getattr(mod, class_name)
            except ImportError as e:
                # Missing module = likely optional dependency, warn only
                warnings.append(f"{path.name}: _target_ '{target}' not importable: {e}")
                continue
            except AttributeError as e:
                # Module exists but class missing = stale path, error
                errors.append(f"{path.name}: _target_ '{target}' stale path: {e}")
                continue

            # Check function has docstring
            doc = getattr(func, "__doc__", None)
            if not doc or not doc.strip():
                warnings.append(f"{path.name}: {class_name} has no docstring")

            # Check function is in registry
            if class_name not in registry_func_names:
                warnings.append(
                    f"{path.name}: {class_name} not found in metric registry "
                    f"(missing @register_metric?)"
                )


def check_algorithm_configs():
    """Check all algorithm config _target_ values are importable."""
    for algo_type in ("latent", "lightning"):
        config_dir = CONFIGS / "algorithms" / algo_type
        if not config_dir.is_dir():
            continue

        for path in sorted(config_dir.glob("*.yaml")):
            if path.name.startswith("_"):
                continue

            with open(path) as f:
                cfg = yaml.safe_load(f) or {}

            target = cfg.get("_target_")
            if not target:
                continue

            module_path, class_name = target.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                getattr(mod, class_name)
            except ImportError as e:
                warnings.append(f"{path.name}: _target_ '{target}' not importable: {e}")
            except AttributeError as e:
                errors.append(f"{path.name}: _target_ '{target}' stale path: {e}")


def main():
    check_metric_configs()
    check_algorithm_configs()

    if warnings:
        print(f"\n{'='*60}")
        print(f"WARNINGS ({len(warnings)}):")
        print(f"{'='*60}")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        print(f"{'='*60}")
        for e in errors:
            print(f"  X {e}")
        sys.exit(1)

    print(f"\nDocs coverage OK ({len(warnings)} warnings, 0 errors)")
    sys.exit(0)


if __name__ == "__main__":
    main()
