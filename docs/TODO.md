# manyLatents TODO

## Pre-Release Checklist

### Add Lightning Module CLI Test
**Priority**: High (blocking release)
**Status**: Not started

Add a CLI test for Lightning modules (e.g., autoencoder) to validate training works.

**Implementation**:
```yaml
# .github/workflows/build.yml
- name: Test CLI - Lightning Module
  run: |
    source .venv/bin/activate
    python -m manylatents.main \
      algorithms/lightning=ae_reconstruction \
      data=swissroll \
      metrics=test_metric \
      callbacks/embedding=minimal \
      trainer.max_epochs=2 \
      trainer.fast_dev_run=true
```

**Why important**:
- Validates Lightning training loop works
- Tests neural network path (not just LatentModule)
- Ensures `fast_dev_run` flag works correctly

---

## Testing & CI

### Lightning Module Integration Test
**Priority**: Medium
**Status**: Not started

Add integration test for Lightning modules through manyAgents orchestration.

**Test should validate**:
- manyAgents can orchestrate Lightning modules (e.g., autoencoder)
- Training loops work through the adapter
- Neural network embeddings are extracted correctly
- Metrics work with Lightning modules

**Implementation**:
```bash
python -m manyagents.main \
  experiment=manylatents_single_algorithm \
  +workflow.steps.0.config.algorithms.lightning=ae_reconstruction \
  +workflow.steps.0.config.trainer.max_epochs=2 \
  +workflow.steps.0.config.metrics=test_metric \
  +workflow.steps.0.config.callbacks.embedding=minimal
```

### Module Instantiation Sweep Test
**Priority**: High
**Status**: Not started

Add pytest that sweeps through all algorithm modules and validates they can be instantiated.

**Goal**: Ensure all modules in `manylatents/algorithms/` can be imported and initialized without errors.

**Implementation**:
```python
# tests/test_module_instantiation.py
import pytest
from pathlib import Path
import importlib

def test_all_algorithms_can_instantiate():
    """Test that all algorithm modules can be imported and instantiated."""

    # Collect all Python files in algorithms/
    algo_dir = Path("manylatents/algorithms")

    for module_file in algo_dir.rglob("*.py"):
        if module_file.name.startswith("_") or module_file.name.startswith("test_"):
            continue

        # Convert path to module name
        module_path = str(module_file.relative_to(".")).replace("/", ".")[:-3]

        # Import and check for classes
        module = importlib.import_module(module_path)

        # Find classes that inherit from LatentModule or LightningModule
        # Try to instantiate with minimal config
        # Assert no import errors or instantiation failures
```

**Benefits**:
- Catches import errors early
- Validates all modules are properly structured
- Prevents broken algorithms from being merged
- Fast smoke test for entire algorithm library

## Future Enhancements

### Domain Extensions Package (`manylatents-domains`)
**Priority**: Low (future architecture)
**Status**: Design phase

**Goal**: Extract domain-specific code into a separate plugin package that houses multiple fields.

**Proposed structure**:
```
manylatents/              # Core framework (algorithms, metrics, base classes)
manylatents-domains/      # Domain-specific extensions (separate package)
  ├── genomics/
  │   ├── datasets/      # HGDP, UKBB, AOU
  │   ├── metrics/       # Admixture preservation, geographic metrics
  │   └── preprocessing/
  ├── singlecell/
  │   ├── datasets/      # scRNA-seq loaders
  │   ├── metrics/       # Cell type separation, trajectory metrics
  │   └── preprocessing/
  ├── vision/
  │   ├── datasets/      # ImageNet, CIFAR, custom vision
  │   ├── metrics/       # Image-specific quality metrics
  │   └── preprocessing/
  └── timeseries/
      ├── datasets/
      ├── metrics/
      └── preprocessing/
```

**Benefits**:
- ✅ **Lighter core**: manylatents stays focused on DR/ML algorithms
- ✅ **Optional install**: `pip install manylatents-domains[genomics]` for specific domains
- ✅ **One plugin repo**: Easier to maintain than per-domain packages
- ✅ **Community contributions**: Domain experts can add their field
- ✅ **Scalable**: Add vision, NLP, timeseries, etc. as needed

**Installation**:
```bash
# Core only
pip install manylatents

# With specific domains
pip install manylatents-domains[genomics]
pip install manylatents-domains[genomics,vision]

# All domains
pip install manylatents-domains[all]
```

**Migration path**:
1. Keep current code in main for now
2. When ready, extract to `manylatents-domains`
3. Maintain compatibility via import redirects
4. Eventually deprecate domain code in main

**When to do this**:
- When core becomes bloated with domain dependencies
- When external contributors want to add new domains
- Before v2.0 release

---

### Logging Config Group
**Priority**: Medium (documented in CLAUDE.md)
**Status**: ✅ COMPLETED (2025-10-15)

Implemented `logger` config group to control wandb initialization:
- `logger=none` - No wandb, fastest (default for CI)
- `logger=wandb` - Full wandb integration

**Changes made:**
- Created `manylatents/configs/logger/` with `none.yaml` and `wandb.yaml`
- Added `logger: Optional[Any] = None` to Config dataclass
- Refactored `experiment.py` to conditionally initialize wandb
- Updated CI workflow to use `logger=none` instead of env vars
- Documented in CLAUDE.md

This eliminates wandb authentication panics in CI and provides clean config-based control.
