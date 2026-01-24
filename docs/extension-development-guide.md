# Developing manylatents Extensions

This guide documents how to create extension packages for manylatents, following the patterns established by `manylatents-omics`. Extensions integrate seamlessly through Python's namespace package system and Hydra's config composition.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Package Structure](#package-structure)
3. [Hydra Config Integration](#hydra-config-integration)
4. [Entry Points](#entry-points)
5. [Component Types](#component-types)
6. [CI Requirements](#ci-requirements)
7. [Testing Checklist](#testing-checklist)

---

## Architecture Overview

manylatents uses a layered extension architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
├─────────────────────────────────────────────────────────────┤
│  manylatents-yourextension   │   manylatents-omics          │
│  (your namespace package)    │   (popgen, dogma, singlecell)│
├─────────────────────────────────────────────────────────────┤
│                        shop (optional)                       │
│              (shared SLURM launchers, logging utils)         │
├─────────────────────────────────────────────────────────────┤
│                      manylatents (core)                      │
│         (LatentModule, metrics, data, experiment runner)     │
└─────────────────────────────────────────────────────────────┘
```

**Key principles:**
- Extensions add domain-specific algorithms, data loaders, and metrics
- Hydra configs from extensions compose with core configs automatically
- Namespace packages allow seamless `from manylatents.yourext import ...` imports

---

## Package Structure

### Directory Layout

```
manylatents-yourextension/
├── pyproject.toml
├── README.md
├── CLAUDE.md                    # AI assistant instructions
├── manylatents/
│   ├── __init__.py              # Namespace package declaration (CRITICAL)
│   ├── yourext_plugin.py        # Hydra SearchPathPlugin
│   └── yourext/
│       ├── __init__.py
│       ├── algorithms/          # Custom LatentModule implementations
│       │   ├── __init__.py
│       │   └── your_algorithm.py
│       ├── data/                # Dataset classes
│       │   ├── __init__.py
│       │   └── your_dataset.py
│       ├── metrics/             # Custom metrics
│       │   ├── __init__.py
│       │   └── your_metric.py
│       └── configs/             # Hydra config files
│           ├── __init__.py      # Empty, but required for pkg://
│           ├── data/
│           │   └── your_data.yaml
│           ├── algorithms/
│           │   └── latent/
│           │       └── your_algo.yaml
│           ├── metrics/
│           │   └── dataset/
│           │       └── your_metric.yaml
│           └── experiment/
│               └── your_experiment.yaml
└── tests/
    ├── __init__.py
    ├── test_imports.py          # Namespace package verification
    └── test_config_e2e.py       # Config resolution tests
```

### Critical File: `manylatents/__init__.py`

This file MUST contain the namespace package declaration:

```python
# Namespace package - allows manylatents.yourext to extend manylatents
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

**Without this, Python won't merge your package with core manylatents.**

---

## Hydra Config Integration

### SearchPathPlugin

Create `manylatents/yourext_plugin.py`:

```python
"""Auto-discover extension configs when package is installed."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class YourExtSearchPathPlugin(SearchPathPlugin):
    """Add extension config packages to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Add core manylatents configs (always present as dependency)
        import manylatents.configs
        search_path.append(
            provider="manylatents",
            path="pkg://manylatents.configs",
        )

        # Add YOUR extension configs with HIGHER priority (prepend)
        # This allows overriding core configs with extension-specific versions
        search_path.prepend(
            provider="manylatents-yourext",
            path="pkg://manylatents.yourext.configs",
        )
```

**Priority matters:** Use `prepend()` if your configs should override core configs with the same name, `append()` if core should take precedence.

### Entry Point Registration

In `pyproject.toml`:

```toml
[project.entry-points."hydra.searchpath"]
manylatents-yourext = "manylatents.yourext_plugin:YourExtSearchPathPlugin"
```

**Important:** Hydra 1.3 doesn't reliably discover entry-point plugins. You must also auto-register in your package's `__init__.py` or provide an alternative entry point.

### Alternative Entry Point (Recommended)

Create `manylatents/yourext/main.py` as an alternative entry point:

```python
"""Entry point for manylatents with yourext configs.

Usage:
    python -m manylatents.yourext.main experiment=your_experiment
"""

# Register SearchPathPlugin BEFORE importing manylatents.main
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from manylatents.yourext_plugin import YourExtSearchPathPlugin

plugins = Plugins.instance()
existing = list(plugins.discover(SearchPathPlugin))
if YourExtSearchPathPlugin not in existing:
    plugins.register(YourExtSearchPathPlugin)

# Now import and run the main function
from manylatents.main import main

if __name__ == "__main__":
    main()
```

Users can then run:
```bash
python -m manylatents.yourext.main experiment=your_experiment
```

---

## Entry Points

### pyproject.toml Configuration

```toml
[project]
name = "manylatents-yourext"
version = "0.1.0"
description = "Your domain extension for manylatents"
requires-python = ">=3.10, <3.13"

dependencies = [
    "manylatents",  # Always depend on core
    # Your domain-specific deps
]

[project.optional-dependencies]
# Heavy deps that not all users need
gpu = ["torch>=2.3"]

[project.entry-points."hydra.searchpath"]
manylatents-yourext = "manylatents.yourext_plugin:YourExtSearchPathPlugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["manylatents"]  # CRITICAL: Package the manylatents/ directory

[tool.uv]
managed = true

[tool.uv.sources]
# During development, point to local or git manylatents
manylatents = { git = "https://github.com/latent-reasoning-works/manylatents.git" }

# Inherit dependency metadata overrides from manylatents
[[tool.uv.dependency-metadata]]
name = "heatgeo"
version = "0.0.1"
requires-dist = ["numpy>=1.25"]
```

### shop Integration (Optional)

If you use `shop` for SLURM launchers:

```toml
[project.optional-dependencies]
slurm = ["shop"]
```

In your main module:
```python
# Optional shop integration for SLURM and dynamic config discovery
try:
    import shop
    from shop.hydra import register_dynamic_search_path
    register_dynamic_search_path()
except ImportError:
    pass  # shop not installed
```

This enables the `HYDRA_SEARCH_PACKAGES` environment variable for dynamic config composition.

---

## Component Types

### Custom LatentModule (Algorithm)

```python
# manylatents/yourext/algorithms/your_algorithm.py

from torch import Tensor
from manylatents.algorithms.latent.latent_module_base import LatentModule


class YourAlgorithm(LatentModule):
    """Your dimensionality reduction algorithm."""

    def __init__(
        self,
        n_components: int = 2,
        your_param: float = 1.0,
        **kwargs
    ):
        super().__init__(n_components=n_components, **kwargs)
        self.your_param = your_param
        # Initialize your model here

    def fit(self, x: Tensor) -> None:
        """Fit the model to data."""
        x_np = x.detach().cpu().numpy()
        # Your fitting logic
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transform data to embedding space."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        x_np = x.detach().cpu().numpy()
        # Your transform logic
        embedding = ...  # Your embedding computation
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    # Optional: expose kernel/affinity matrices for module metrics
    def kernel_matrix(self, ignore_diagonal: bool = False):
        """Return the kernel matrix (if applicable)."""
        raise NotImplementedError("YourAlgorithm doesn't use a kernel")
```

**Hydra config:** `manylatents/yourext/configs/algorithms/latent/your_algo.yaml`

```yaml
_target_: manylatents.yourext.algorithms.YourAlgorithm
n_components: 2
your_param: 1.0
```

### Custom Dataset

```python
# manylatents/yourext/data/your_dataset.py

import numpy as np
from typing import Optional


class YourDataset:
    """Your domain-specific dataset."""

    def __init__(
        self,
        data_path: str,
        n_samples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.n_samples = n_samples
        self._data = None
        self._labels = None
        self._load_data()

    def _load_data(self):
        """Load and preprocess data."""
        # Your loading logic
        self._data = np.load(self.data_path)
        if self.n_samples:
            self._data = self._data[:self.n_samples]

    @property
    def data(self) -> np.ndarray:
        """High-dimensional data array."""
        return self._data

    def get_data(self) -> np.ndarray:
        """Get data (interface for auto-discovery)."""
        return self._data

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Optional labels for evaluation."""
        return self._labels
```

**Hydra config:** `manylatents/yourext/configs/data/your_data.yaml`

```yaml
_target_: manylatents.yourext.data.YourDataset
data_path: ${paths.data_dir}/your_data.npy
n_samples: null
```

### Custom Metric

```python
# manylatents/yourext/metrics/your_metric.py

from typing import Optional
import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


@register_metric(
    aliases=["your_metric", "ym"],
    default_params={"threshold": 0.5},
    description="Your domain-specific embedding quality metric",
)
def YourMetric(
    embeddings: np.ndarray,
    dataset: object,
    module: Optional[LatentModule] = None,
    threshold: float = 0.5,
    return_per_sample: bool = False,
) -> float:
    """
    Compute your metric on embeddings.

    Args:
        embeddings: Low-dimensional embedding array (N, D).
        dataset: Dataset object with .data attribute (high-dim data).
        module: Optional LatentModule for accessing model internals.
        threshold: Your metric parameter.
        return_per_sample: If True, return (mean, per_sample_array).

    Returns:
        Metric score (float) or tuple (mean, per_sample) if return_per_sample.
    """
    # Your metric computation
    scores = ...  # Per-sample scores

    if return_per_sample:
        return float(np.mean(scores)), scores
    return float(np.mean(scores))
```

**Hydra config:** `manylatents/yourext/configs/metrics/dataset/your_metric.yaml`

```yaml
_target_: manylatents.yourext.metrics.YourMetric
_partial_: true  # CRITICAL: Creates functools.partial for deferred binding
threshold: 0.5
return_per_sample: false
```

### Experiment Config

`manylatents/yourext/configs/experiment/your_experiment.yaml`:

```yaml
# @package _global_
name: your_experiment
project: your_project

defaults:
  - override /algorithms/latent: your_algo  # From YOUR extension
  - override /data: your_data               # From YOUR extension
  - override /callbacks/embedding: default  # From CORE manylatents
  - override /metrics: default              # From CORE manylatents

seed: 42

# Override algorithm params
algorithms:
  latent:
    n_components: 10
    your_param: 2.0

# Add extension-specific metrics
metrics:
  dataset:
    your_metric:
      _target_: manylatents.yourext.metrics.YourMetric
      _partial_: true
      threshold: 0.3
```

---

## CI Requirements

### Required CI Tests

Your extension must pass these tests to ensure compatibility:

#### 1. Import Tests (`tests/test_imports.py`)

```python
"""Test that extension modules can be imported correctly."""

import pytest


def test_extension_package_import():
    """Test that the extension package can be imported."""
    import manylatents.yourext
    assert hasattr(manylatents.yourext, '__version__')


def test_algorithm_imports():
    """Test that all algorithm classes can be imported."""
    from manylatents.yourext.algorithms import YourAlgorithm
    assert YourAlgorithm is not None


def test_data_imports():
    """Test that all dataset classes can be imported."""
    from manylatents.yourext.data import YourDataset
    assert YourDataset is not None


def test_metric_imports():
    """Test that all metrics can be imported."""
    from manylatents.yourext.metrics import YourMetric
    assert YourMetric is not None


def test_namespace_package_structure():
    """Test namespace package is set up correctly."""
    import manylatents.yourext
    assert hasattr(manylatents.yourext, 'algorithms')
    assert hasattr(manylatents.yourext, 'data')


def test_core_manylatents_accessible():
    """Test that core manylatents can still be imported."""
    from manylatents.data.synthetic_dataset import SwissRoll
    from manylatents.algorithms.latent import PCAModule
    import manylatents.metrics.trustworthiness

    assert SwissRoll is not None
    assert PCAModule is not None
```

#### 2. Config Resolution Tests (`tests/test_config_e2e.py`)

```python
"""E2E config tests - validate Hydra configs resolve correctly."""

import pytest
from omegaconf import OmegaConf
from pathlib import Path


CONFIGS_DIR = Path(__file__).parent.parent / "manylatents" / "yourext" / "configs"


class TestConfigResolution:
    """Test config files resolve without errors."""

    def test_algorithm_config_loads(self):
        """Test algorithm config has valid _target_."""
        config_path = CONFIGS_DIR / "algorithms" / "latent" / "your_algo.yaml"
        cfg = OmegaConf.load(config_path)
        assert "_target_" in cfg
        assert cfg._target_.startswith("manylatents.yourext")

    def test_data_config_loads(self):
        """Test data config has valid _target_."""
        config_path = CONFIGS_DIR / "data" / "your_data.yaml"
        cfg = OmegaConf.load(config_path)
        assert "_target_" in cfg

    def test_experiment_config_loads(self):
        """Test experiment config loads and has required fields."""
        config_path = CONFIGS_DIR / "experiment" / "your_experiment.yaml"
        cfg = OmegaConf.load(config_path)
        assert cfg.name is not None
        assert cfg.project is not None


class TestTargetPaths:
    """Test that _target_ paths are valid Python imports."""

    def test_all_targets_are_importable(self):
        """Verify _target_ values point to existing classes."""
        for config_file in CONFIGS_DIR.rglob("*.yaml"):
            cfg = OmegaConf.load(config_file)
            if hasattr(cfg, "_target_"):
                target = cfg._target_
                assert "." in target, f"{config_file}: invalid target"
                assert target.startswith("manylatents"), f"{config_file}: should start with manylatents"
```

#### 3. Integration Tests

```python
"""Integration tests with core manylatents."""

import pytest


def test_algorithm_interface_compliance():
    """Test that YourAlgorithm implements LatentModule interface."""
    from manylatents.yourext.algorithms import YourAlgorithm
    import torch

    algo = YourAlgorithm(n_components=2)

    # Test interface methods exist
    assert hasattr(algo, 'fit')
    assert hasattr(algo, 'transform')
    assert hasattr(algo, 'fit_transform')

    # Test with synthetic data
    X = torch.randn(100, 50)
    embedding = algo.fit_transform(X)

    assert embedding.shape == (100, 2)
    assert algo._is_fitted


def test_metric_signature():
    """Test that YourMetric has correct signature."""
    from manylatents.yourext.metrics import YourMetric
    import inspect

    sig = inspect.signature(YourMetric)
    params = list(sig.parameters.keys())

    # Required parameters
    assert 'embeddings' in params
    assert 'dataset' in params
    assert 'module' in params
```

### GitHub Actions CI Workflow

`.github/workflows/ci.yml`:

```yaml
name: CI - Integration with manylatents

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-integration:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - name: Checkout extension
      uses: actions/checkout@v4
      with:
        path: manylatents-yourext

    - name: Checkout manylatents (main)
      uses: actions/checkout@v4
      with:
        repository: latent-reasoning-works/manylatents
        ref: main
        path: manylatents
        token: ${{ secrets.GITHUB_TOKEN }}

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      uses: astral-sh/setup-uv@v5
      with:
        enable-cache: true

    - name: Install manylatents (core)
      working-directory: manylatents
      run: uv sync

    - name: Install extension
      working-directory: manylatents-yourext
      run: |
        uv venv
        uv pip install -e ../manylatents
        uv pip install -e .

    - name: Run extension tests
      working-directory: manylatents-yourext
      run: uv run pytest tests/ -v

    - name: Test namespace package imports
      working-directory: manylatents-yourext
      run: |
        uv run python -c "
        # Test extension imports
        from manylatents.yourext.algorithms import YourAlgorithm
        from manylatents.yourext.data import YourDataset
        print('Extension imports OK')

        # Test core manylatents still works
        from manylatents.data.synthetic_dataset import SwissRoll
        from manylatents.algorithms.latent import PCAModule
        print('Core manylatents imports OK')

        # Verify namespace structure
        import manylatents
        print(f'manylatents.__path__: {manylatents.__path__}')
        assert len(manylatents.__path__) >= 2, 'Namespace not merged'
        print('Namespace package structure OK')
        "

    - name: Run core manylatents tests (smoke test)
      working-directory: manylatents
      run: |
        uv run pytest manylatents/data/test_data.py -v
        echo "Core tests still pass with extension installed"
```

---

## Testing Checklist

Before publishing your extension, verify:

### Namespace Package
- [ ] `manylatents/__init__.py` contains `__path__ = __import__('pkgutil').extend_path(__path__, __name__)`
- [ ] `import manylatents.yourext` works
- [ ] `from manylatents.yourext.algorithms import YourAlgorithm` works
- [ ] Core manylatents still importable: `from manylatents.data import SwissRoll`

### Hydra Config
- [ ] SearchPathPlugin is registered (entry-point + manual registration)
- [ ] All configs have valid `_target_` paths
- [ ] Metrics use `_partial_: true`
- [ ] Experiment configs use `# @package _global_` directive
- [ ] `python -m manylatents.yourext.main experiment=your_experiment` works

### Interface Compliance
- [ ] LatentModule subclasses implement `fit()` and `transform()`
- [ ] Datasets have `data` property and `get_data()` method
- [ ] Metrics accept `(embeddings, dataset, module, **kwargs)` signature

### CI
- [ ] Import tests pass on Python 3.10, 3.11, 3.12
- [ ] Config resolution tests pass
- [ ] Core manylatents tests still pass when extension is installed
- [ ] No circular import issues

### Documentation
- [ ] README.md with installation instructions
- [ ] CLAUDE.md for AI assistant context
- [ ] Example experiment configs
- [ ] API documentation for custom components

---

## Quick Reference

### Running with Extension Configs

```bash
# Using extension's alternative entry point (recommended)
python -m manylatents.yourext.main experiment=your_experiment

# Using environment variable (requires shop)
HYDRA_SEARCH_PACKAGES="manylatents.configs:manylatents.yourext.configs" \
    python -m manylatents.main experiment=your_experiment

# Explicit config path (fallback)
python -m manylatents.main \
    --config-path=/path/to/manylatents-yourext/manylatents/yourext/configs \
    experiment=your_experiment
```

### Development Workflow

```bash
# From extension repo (recommended for extension development)
cd manylatents-yourext
uv sync  # Pulls manylatents from git
uv run python -m manylatents.yourext.main experiment=your_experiment

# From manylatents repo (for testing integration)
cd manylatents
uv add git+https://github.com/you/manylatents-yourext.git
uv run python -m manylatents.main experiment=your_experiment
```

### Debugging Config Discovery

```python
# Check which plugins are registered
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

plugins = list(Plugins.instance().discover(SearchPathPlugin))
print([p.__name__ for p in plugins])
# Should show: ['ManylatentsSearchPathPlugin', 'YourExtSearchPathPlugin']
```
