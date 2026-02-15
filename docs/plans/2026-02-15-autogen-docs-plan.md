# Auto-Generated Docs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hand-maintained tables in docs with auto-generated tables from Hydra configs and the metric registry, enforced by CI.

**Architecture:** `mkdocs-macros-plugin` provides Jinja2 macros inside markdown. A `docs/macros.py` hook walks YAML configs and the `@register_metric` registry at build time, exposing `{{ algorithm_table() }}`, `{{ metrics_table() }}`, etc. CI runs `mkdocs build --strict` and a coverage checker on every push/PR. Deploy to GitHub Pages on main.

**Tech Stack:** mkdocs-material (already installed), mkdocs-macros-plugin (new), PyYAML (already available), GitHub Actions

**Base path:** `/network/scratch/c/cesar.valdez/mbyl_for_practitioners/experiments/tools/manylatents/`

---

### Task 1: Add mkdocs-macros-plugin dependency

**Files:**
- Modify: `pyproject.toml:55-57`

**Step 1: Add dependency**

In `pyproject.toml`, find this block:

```toml
    "mkdocs>=1.6",
    "mkdocs-material>=9.6",
```

Change to:

```toml
    "mkdocs>=1.6",
    "mkdocs-material>=9.6",
    "mkdocs-macros-plugin>=1.0",
```

**Step 2: Sync dependencies**

```bash
uv sync
```

Expected: resolves and installs `mkdocs-macros-plugin`. No conflicts.

**Step 3: Verify import**

```bash
uv run python -c "import mkdocs_macros; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add mkdocs-macros-plugin dependency"
```

---

### Task 2: Create docs/macros.py with four table macros

**Files:**
- Create: `docs/macros.py`
- Create: `tests/test_docs_macros.py`

**Step 1: Write tests**

Create `tests/test_docs_macros.py`:

```python
"""Tests for docs/macros.py auto-gen table functions."""
import sys
import os
import pytest

# Add docs/ to path so we can import macros
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs"))
import macros


def test_algorithm_table_latent():
    """algorithm_table('latent') returns a markdown table with known algorithms."""
    table = macros._algorithm_table("latent")
    assert "| algorithm |" in table.lower() or "| Algorithm |" in table
    assert "PCA" in table
    assert "UMAP" in table
    assert "`algorithms/latent=pca`" in table


def test_algorithm_table_lightning():
    """algorithm_table('lightning') returns lightning algorithms."""
    table = macros._algorithm_table("lightning")
    assert "ae_reconstruction" in table or "Reconstruction" in table


def test_metrics_table_embedding():
    """metrics_table('embedding') returns embedding metrics."""
    # Trigger metric registration
    import manylatents.metrics  # noqa: F401

    table = macros._metrics_table("embedding")
    assert "trustworthiness" in table.lower()
    assert "`metrics/embedding=" in table


def test_metrics_table_module():
    """metrics_table('module') returns module metrics."""
    import manylatents.metrics  # noqa: F401

    table = macros._metrics_table("module")
    assert "affinity_spectrum" in table.lower() or "AffinitySpectrum" in table


def test_data_table():
    """data_table() returns data modules."""
    table = macros._data_table()
    assert "swissroll" in table.lower()
    assert "`data=" in table


def test_sampling_table():
    """sampling_table() returns sampling strategies."""
    table = macros._sampling_table()
    assert "random" in table.lower() or "Random" in table


def test_tables_are_valid_markdown():
    """All tables should have header separator row."""
    import manylatents.metrics  # noqa: F401

    for fn in [
        lambda: macros._algorithm_table("latent"),
        lambda: macros._metrics_table("embedding"),
        lambda: macros._data_table(),
        lambda: macros._sampling_table(),
    ]:
        table = fn()
        lines = table.strip().split("\n")
        assert len(lines) >= 3, f"Table too short: {table[:100]}"
        assert "---" in lines[1], f"No separator row: {lines[1]}"
```

**Step 2: Run tests — expect FAIL**

```bash
uv run pytest tests/test_docs_macros.py -v
```

Expected: `ModuleNotFoundError: No module named 'macros'`

**Step 3: Create `docs/macros.py`**

```python
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
        # Map function name -> description (from alias entries which have descriptions)
        descs = {}
        for alias, spec in registry.items():
            func_name = spec.func.__name__
            desc = spec.description.strip().split("\n")[0] if spec.description else ""
            # Prefer the alias description over the raw docstring
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
        # Find the first key that has a _target_
        inner = None
        metric_key = config_name
        for key, val in cfg.items():
            if isinstance(val, dict) and "_target_" in val:
                inner = val
                metric_key = key
                break

        if inner is None:
            # Flat config (shouldn't happen for metrics, but handle gracefully)
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
        # Truncate long descriptions
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

    # Skip internal/test configs
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

        # Extract a few interesting params
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
        # Sampling configs may be nested under 'sampling' key
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
```

**Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_docs_macros.py -v
```

**Step 5: Commit**

```bash
git add docs/macros.py tests/test_docs_macros.py
git commit -m "feat: add docs/macros.py with auto-gen table functions"
```

---

### Task 3: Configure mkdocs-macros in mkdocs.yml

**Files:**
- Modify: `mkdocs.yml`

**Step 1: Update mkdocs.yml**

Replace the entire file with:

```yaml
site_name: manylatents
site_url: https://latent-reasoning-works.github.io/manylatents/

theme:
  name: material

plugins:
  - search
  - macros:
      module_name: docs/macros

markdown_extensions:
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.superfences
  - pymdownx.highlight
  - tables
  - admonition
  - pymdownx.tasklist:
      custom_checkbox: true

nav:
  - Home: index.md
  - Algorithms: algorithms.md
  - Data: data.md
  - Metrics: metrics.md
  - Evaluation: evaluation.md
  - Cache Protocol: cache.md
  - Callbacks: callbacks.md
  - Extensions: extensions.md
  - API: api_usage.md
  - Probing: probing.md
  - Testing: testing.md
```

Changes from original:
- `site_name` lowercase to match README style
- Added `site_url` for GitHub Pages
- Added `plugins` section with `macros` (module_name points to `docs/macros`)
- Added `search` plugin (Material default, must be explicit when adding other plugins)
- Added `Data` and `Cache Protocol` to nav

**Step 2: Verify mkdocs loads the plugin**

```bash
uv run mkdocs build 2>&1 | head -20
```

Expected: no errors about macros plugin. May warn about missing `data.md` and `cache.md` pages (we'll create those next).

**Step 3: Commit**

```bash
git add mkdocs.yml
git commit -m "config: add mkdocs-macros plugin to mkdocs.yml"
```

---

### Task 4: Create new docs pages (data.md, cache.md)

**Files:**
- Create: `docs/data.md`
- Create: `docs/cache.md`

**Step 1: Create `docs/data.md`**

```markdown
# Data

manyLatents provides synthetic manifold datasets for benchmarking and a precomputed loader for custom data.

{{ data_table() }}

Domain-specific datasets (genomics, single-cell) are available via the [manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics) extension.

## Precomputed Data

Load your own data from `.npy` or `.npz` files:

```bash
python -m manylatents.main data=precomputed data.path=/path/to/data.npy algorithms/latent=umap
```

## Sampling

Large datasets are subsampled before metric evaluation. Configure under `metrics.sampling`:

{{ sampling_table() }}
```

**Step 2: Create `docs/cache.md`**

```markdown
# Cache Protocol

All metrics share a single `cache` dict. The config sleuther discovers k-values from metric configs and pre-warms kNN and eigenvalues before any metric runs.

```python
# this happens automatically inside evaluate_embeddings()
cache = {}
compute_knn(high_dim_data, k=25, cache=cache)    # computed once
compute_knn(embeddings,     k=25, cache=cache)    # computed once
compute_eigenvalues(module, cache=cache)           # computed once

# every metric reuses the same cache
trustworthiness(emb, dataset=ds, cache=cache)
continuity(emb, dataset=ds, cache=cache)
```

`compute_knn` selects the fastest available backend: FAISS-GPU > FAISS-CPU > sklearn.

## How It Works

`evaluate_embeddings()` uses `extract_k_requirements()` to discover all `k`/`n_neighbors` values from metric configs, then calls `prewarm_cache()` to compute kNN once with `max(k)`:

1. **Sleuther** extracts requirements from metric configs
2. **Pre-warm** computes kNN and eigenvalues at optimal k values
3. **Metrics** receive the shared cache — `compute_knn()` slices cached results for smaller k values

## Extension Metrics

Extension metrics that don't accept `cache=` are handled gracefully via a `TypeError` fallback. A warning is logged suggesting the extension add `cache=None` to its signature.
```

**Step 3: Verify build**

```bash
uv run mkdocs build 2>&1 | tail -5
```

Expected: clean build, no errors.

**Step 4: Commit**

```bash
git add docs/data.md docs/cache.md
git commit -m "docs: add data.md and cache.md pages"
```

---

### Task 5: Replace hand-written tables with macro calls in existing pages

**Files:**
- Modify: `docs/algorithms.md`
- Modify: `docs/metrics.md`
- Modify: `docs/index.md`

**Step 1: Update `docs/algorithms.md`**

Find the hand-written LatentModule table (inside the `=== "LatentModule"` tab). Replace this block:

```markdown
    ### Available Algorithms

    | Algorithm | Class | Config | Key Parameters |
    |-----------|-------|--------|----------------|
    | PCA | `PCAModule` | `algorithms/latent=pca` | `n_components`, `random_state` |
    | t-SNE | `TSNEModule` | `algorithms/latent=tsne` | `n_components`, `perplexity`, `learning_rate`, `metric` |
    | UMAP | `UMAPModule` | `algorithms/latent=umap` | `n_components`, `n_neighbors`, `min_dist`, `metric` |
    | PHATE | `PHATEModule` | `algorithms/latent=phate` | `n_components`, `knn`, `gamma` |
    | MultiscalePHATE | `MultiscalePHATEModule` | `algorithms/latent=multiscale_phate` | `n_components`, `knn` |
    | DiffusionMap | `DiffusionMapModule` | `algorithms/latent=diffusionmap` | `n_components`, `knn`, `decay` |
    | MDS | `MDSModule` | `algorithms/latent=mds` | `n_components` |
    | Archetypes | `AAModule` | `algorithms/latent=aa` | `n_components` |
    | Classifier | `ClassifierModule` | `algorithms/latent=classifier` | Supervised; uses labels |
    | Noop | `DRNoop` | `algorithms/latent=noop` | Passthrough (identity) |
    | MergingModule | `MergingModule` | `algorithms/latent=merging` | `strategy`, `target_dim` |
```

With:

```markdown
    ### Available Algorithms

    {{ algorithm_table("latent") }}
```

Find the LightningModule table (inside the `=== "LightningModule"` tab). Replace this block:

```markdown
    ### Available Algorithms

    | Algorithm | Class | Config | Description |
    |-----------|-------|--------|-------------|
    | Reconstruction | `Reconstruction` | `algorithms/lightning=ae_reconstruction` | Autoencoder reconstruction |
    | AANet Reconstruction | (uses Reconstruction) | `algorithms/lightning=aanet_reconstruction` | Archetypal network |
    | LatentODE | `LatentODE` | `algorithms/lightning=latent_ode` | Neural ODE in latent space |
    | HF Trainer | `HFTrainerModule` | `algorithms/lightning=hf_trainer` | HuggingFace model training |
```

With:

```markdown
    ### Available Algorithms

    {{ algorithm_table("lightning") }}
```

**Step 2: Update `docs/metrics.md`**

In the `=== "Architecture"` tab, find and replace the three hand-written metric lists.

The current metrics.md doesn't have explicit tables for each context — it describes them in prose. That's fine; the macro calls add the actual reference tables. Add macro calls after the descriptions of each level:

After the `### 2. Embedding Metrics` description block, add:

```markdown
    {{ metrics_table("embedding") }}
```

After the `### 3. Module Metrics` description block, add:

```markdown
    {{ metrics_table("module") }}
```

After the `### 1. Dataset Metrics` description block, add:

```markdown
    {{ metrics_table("dataset") }}
```

**Step 3: Update `docs/index.md`**

Remove the LRW Ecosystem table (sequential releases — this is the first). Replace the entire `## LRW Ecosystem` section at the bottom with nothing (delete it). Also update "Part of the Latent Reasoning Works ecosystem" to just "Dimensionality reduction and neural network analysis."

**Step 4: Verify build**

```bash
uv run mkdocs build --strict 2>&1 | tail -10
```

Expected: clean build. The macro calls should render as tables.

**Step 5: Spot-check rendered output**

```bash
uv run python -c "
from pathlib import Path
html = Path('site/algorithms/index.html').read_text()
assert 'PCA' in html, 'PCA not in algorithms page'
assert 'UMAP' in html, 'UMAP not in algorithms page'
print('algorithms page OK')

html = Path('site/metrics/index.html').read_text()
assert 'trustworthiness' in html.lower(), 'trustworthiness not in metrics page'
print('metrics page OK')

html = Path('site/data/index.html').read_text()
assert 'swissroll' in html.lower(), 'swissroll not in data page'
print('data page OK')
"
```

**Step 6: Commit**

```bash
git add docs/algorithms.md docs/metrics.md docs/index.md
git commit -m "docs: replace hand-written tables with auto-gen macro calls"
```

---

### Task 6: Create docs coverage checker

**Files:**
- Create: `scripts/check_docs_coverage.py`
- Create: `tests/test_docs_coverage.py`

**Step 1: Write test**

Create `tests/test_docs_coverage.py`:

```python
"""Test that check_docs_coverage.py catches problems."""
import subprocess
import sys


def test_coverage_script_runs():
    """The coverage checker should exit 0 on current codebase."""
    result = subprocess.run(
        [sys.executable, "scripts/check_docs_coverage.py"],
        capture_output=True, text=True
    )
    # Should pass (or warn) — not crash
    assert result.returncode == 0, f"Coverage check failed:\n{result.stderr}\n{result.stdout}"
```

**Step 2: Create `scripts/check_docs_coverage.py`**

```python
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
            except (ImportError, AttributeError) as e:
                errors.append(f"{path.name}: _target_ '{target}' not importable: {e}")
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
            except (ImportError, AttributeError) as e:
                errors.append(f"{path.name}: _target_ '{target}' not importable: {e}")


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
```

**Step 3: Run the coverage checker**

```bash
uv run python scripts/check_docs_coverage.py
```

Expected: exits 0. May print warnings for metrics without `@register_metric` (those are known — some are "not yet decorated" per `__init__.py`).

**Step 4: Run test**

```bash
uv run pytest tests/test_docs_coverage.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/check_docs_coverage.py tests/test_docs_coverage.py
git commit -m "ci: add docs coverage checker script"
```

---

### Task 7: Add docs CI jobs to GitHub Actions

**Files:**
- Modify: `.github/workflows/build.yml`

**Step 1: Add docs jobs**

Append the following jobs after the existing `build-and-test` job in `.github/workflows/build.yml`:

```yaml

  # Docs build verification and coverage check
  docs:
    name: Docs Build & Coverage
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install UV package manager
        uses: astral-sh/setup-uv@v5
        with:
          version: 'latest'
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          uv sync

      - name: Check docs coverage
        run: |
          source .venv/bin/activate
          python scripts/check_docs_coverage.py

      - name: Build docs (strict)
        run: |
          source .venv/bin/activate
          mkdocs build --strict

  # Deploy docs to GitHub Pages (main branch only)
  docs-deploy:
    name: Deploy Docs
    needs: [build-and-test, docs]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install UV package manager
        uses: astral-sh/setup-uv@v5
        with:
          version: 'latest'
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          uv sync

      - name: Deploy to GitHub Pages
        run: |
          source .venv/bin/activate
          mkdocs gh-deploy --force
```

**Step 2: Verify YAML syntax**

```bash
uv run python -c "
import yaml
with open('.github/workflows/build.yml') as f:
    cfg = yaml.safe_load(f)
print(f'Jobs: {list(cfg[\"jobs\"].keys())}')
assert 'docs' in cfg['jobs'], 'docs job missing'
assert 'docs-deploy' in cfg['jobs'], 'docs-deploy job missing'
print('CI config OK')
"
```

**Step 3: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "ci: add docs build, coverage check, and GitHub Pages deploy"
```

---

### Task 8: Slim down README

**Files:**
- Modify: `README.md`

**Step 1: Replace the heavy table sections**

Replace the `## algorithms` section (lines ~100-120) — everything from `## algorithms` through the table and the `neighborhood_size` paragraph — with:

```markdown
## algorithms

> 12 algorithms -- 8 latent modules, 4 lightning modules

PCA, t-SNE, UMAP, PHATE, DiffusionMap, MDS, Archetypes, MultiscalePHATE,
Autoencoder, AANet, LatentODE, HF Trainer.

`neighborhood_size=k` sweeps kNN uniformly across algorithms.

[Full reference](https://latent-reasoning-works.github.io/manylatents/algorithms/)
```

Replace the `## metrics` section (lines ~123-183) — everything from `## metrics` through all three sub-tables and the sampling table — with:

```markdown
## metrics

> 20+ metrics across three evaluation contexts

Embedding fidelity (trustworthiness, continuity, kNN preservation), spectral
analysis (affinity spectrum, spectral decay rate), topological features
(persistent homology), and dataset properties (stratification).

Config pattern: `metrics/embedding=<name>`, `metrics/module=<name>`, `metrics/dataset=<name>`

[Full reference](https://latent-reasoning-works.github.io/manylatents/metrics/)
```

Replace the `## data` section (lines ~206-217) with:

```markdown
## data

> 6 synthetic manifolds + precomputed loader

Swiss roll, torus, saddle surface, gaussian blobs, DLA tree, and custom `.npy`/`.npz` files.

[Full reference](https://latent-reasoning-works.github.io/manylatents/data/)
```

Remove the `## extensions` section entirely (lines ~221-229). Extensions info lives in the docs site.

**Step 2: Verify README renders cleanly**

```bash
uv run python -c "
text = open('README.md').read()
assert 'Full reference' in text, 'Missing doc links'
assert '|---|' not in text.split('## quickstart')[1].split('## architecture')[0], 'Tables still in quickstart'
# Tables should only appear in architecture section
print(f'README: {len(text)} chars, {text.count(chr(10))} lines')
print('README OK')
"
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: slim README — replace tables with doc site links"
```

---

### Task 9: Full verification

**Step 1: Run all tests**

```bash
uv run pytest tests/test_docs_macros.py tests/test_docs_coverage.py -v
```

Expected: all pass.

**Step 2: Run coverage checker**

```bash
uv run python scripts/check_docs_coverage.py
```

Expected: exits 0.

**Step 3: Build docs**

```bash
uv run mkdocs build --strict
```

Expected: clean build, no warnings, no errors.

**Step 4: Spot-check the built site**

```bash
uv run python -c "
from pathlib import Path
for page in ['algorithms', 'metrics', 'data']:
    html = Path(f'site/{page}/index.html').read_text()
    assert '<table>' in html, f'{page} page has no table'
    print(f'{page}: OK ({len(html)} chars)')
print('All pages verified')
"
```

**Step 5: Run existing test suite (regression)**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: 189+ pass, 0 fail.

---

## Verification Checklist

After all tasks:

- [ ] `uv run mkdocs build --strict` exits 0
- [ ] `uv run python scripts/check_docs_coverage.py` exits 0
- [ ] `uv run pytest tests/test_docs_macros.py tests/test_docs_coverage.py -v` all pass
- [ ] `uv run pytest tests/ -v --tb=short` no regressions
- [ ] Built site has auto-generated tables on algorithms, metrics, data pages
- [ ] README has no heavy tables — only one-liners with links to docs
- [ ] CI workflow has docs build + coverage + deploy jobs
