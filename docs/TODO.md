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

### Logging Config Group
**Priority**: Medium (documented in CLAUDE.md)
**Status**: Design phase

Replace `debug` flag with proper `logging` config group:
- `logging=none` - No wandb, fastest
- `logging=wandb` - Full wandb integration
- `logging=tensorboard` - TensorBoard support
- `logging=mlflow` - MLflow support

This decouples wandb initialization from verbose logging.
