# ManyLatents TODO

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

---

## Architecture

### Full Non-Lightning Inference Mode
**Priority**: Low (future consideration)
**Status**: Not started

Currently, `FoundationEncoder` (from `manylatents-omics`) uses a workaround: it inherits from `LatentModule` with `fit()` as no-op and `transform()` calling `datamodule.get_sequences()`. This works but has limitations:

**Current workaround** (`manylatents/algorithms/encoder/base.py`):
```python
class FoundationEncoder(LatentModule):
    def fit(self, x: Tensor) -> None:
        self._is_fitted = True  # no-op

    def transform(self, x: Tensor) -> Tensor:
        sequences = self.datamodule.get_sequences()  # ignores x
        return self.encode_batch(sequences)
```

**Limitations**:
- The `x` tensor parameter is ignored (sequences come from datamodule)
- `experiment.py` requires special-casing to skip tensor unrolling for FoundationEncoder
- Doesn't fit cleanly into the fit/transform paradigm (nothing to fit)

**Potential solution**: Add a dedicated `inference` algorithm mode alongside `latent` and `lightning`:
- No fit step, just encode
- Direct datamodule → encoder → embeddings pipeline
- Cleaner separation from dimensionality reduction algorithms

**References**:
- Extension implementation: `manylatents-omics/manylatents/dogma/encoders/`
- SequenceDataModule: `manylatents-omics/manylatents/dogma/data/sequence_dataset.py`
- Experiment special-casing: `manylatents/experiment.py` (FoundationEncoder branch)

---

## Completed

### Logging Config Group
**Priority**: Medium
**Status**: ✅ COMPLETED

Implemented `logger` config group to control wandb initialization:
- `logger=none` - No wandb, fastest (default for CI)
- `logger=wandb` - Full wandb integration

### Centralized Tensor Conversion
**Priority**: Medium
**Status**: ✅ COMPLETED

Moved tensor-to-numpy conversion from individual metrics to `evaluate_embeddings()` in `experiment.py`. Removed duplicate code from 10+ metric files.
