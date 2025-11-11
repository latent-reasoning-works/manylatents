# ManyLatents Logging Fix Summary

## Problem Statement

When Geomancer orchestrates manyLatents subprocesses, WandB logging was not properly disabled, causing:
1. Multiple processes trying to log to the same WandB run
2. WandB conflicts and authentication issues
3. Brittleness in the `debug` flag implementation

The issue was that `logger=None` would disable top-level WandB but NOT the PyTorch Lightning trainer loggers, creating inconsistencies.

## Changes Made

### 1. Created Centralized `should_disable_wandb()` Function

**File**: `manylatents/experiment.py:28-62`

Added a centralized function that checks all three conditions:
- `cfg.logger is None` - Orchestrated by parent (Geomancer)
- `cfg.debug is True` - Fast testing/CI mode
- `WANDB_MODE=disabled` environment variable - External override

This ensures consistent behavior across the entire codebase.

```python
def should_disable_wandb(cfg: DictConfig) -> bool:
    """
    Determine if WandB should be disabled based on configuration.

    WandB is disabled when:
    1. logger is explicitly set to None (orchestrated by parent like Geomancer)
    2. debug mode is True (fast testing/CI)
    3. WANDB_MODE environment variable is set to 'disabled'
    """
    import os

    # Check explicit logger=None (orchestrated mode)
    if cfg.logger is None:
        logger.info("WandB disabled: logger=None (orchestrated by parent)")
        return True

    # Check debug mode
    if cfg.debug:
        logger.info("WandB disabled: debug=True")
        return True

    # Check environment variable (allows external override)
    if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
        logger.info("WandB disabled: WANDB_MODE=disabled")
        return True

    return False
```

### 2. Fixed `run_algorithm()` to Use Centralized Check

**File**: `manylatents/experiment.py:327-380`

Changed from:
```python
if cfg.logger is not None:
    # Initialize wandb...
```

To:
```python
wandb_disabled = should_disable_wandb(cfg)

if not wandb_disabled and cfg.logger is not None:
    # Initialize wandb...
else:
    logger.info("WandB logging disabled - skipping wandb initialization")
```

Also updated trainer logger instantiation:
```python
loggers = []
if not wandb_disabled and cfg.logger is not None:
    for lg_conf in cfg.trainer.get("logger", {}).values():
        loggers.append(hydra.utils.instantiate(lg_conf))
    logger.info(f"Trainer loggers enabled: {len(loggers)} logger(s)")
else:
    logger.info("Trainer loggers disabled (WandB disabled)")
```

### 3. Fixed `run_pipeline()` Hardcoded `wandb.init()`

**File**: `manylatents/experiment.py:474-522`

The critical fix! Previously, `run_pipeline()` had:
```python
with wandb.init(
    project=cfg.project,
    name=cfg.name,
    config=OmegaConf.to_container(cfg, resolve=True),
    mode="disabled" if cfg.debug else "online",  # ❌ Only checked debug, not logger
) as run:
```

Now properly respects all disable conditions:
```python
# Determine if WandB should be disabled using centralized check
wandb_disabled = should_disable_wandb(cfg)

# Initialize WandB based on configuration (respects logger=None, debug=True, and WANDB_MODE)
if wandb_disabled:
    # Use disabled mode - wandb.init() will return a no-op run
    logger.info("Initializing WandB in disabled mode")
    wandb_context = wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled",
    )
else:
    # Normal WandB initialization
    logger.info("Initializing WandB in online mode")
    wandb_context = wandb.init(
        project=cfg.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online",
    )

with wandb_context as run:
    # ... pipeline logic
```

Also fixed trainer logger instantiation in pipeline (line 515-522) to match `run_algorithm()`.

## Testing

Created comprehensive test suite in `test_logging_fix.py`:

### Test Results

✅ **Test 1: logger=None** - PASSED
- Verifies that `logger=None` disables all WandB operations
- Metrics are computed and returned cleanly

✅ **Test 2: debug=True** - PASSED
- Verifies that `debug=True` disables all WandB operations
- Fast testing mode works correctly

✅ **Test 3: WANDB_MODE=disabled** - PASSED
- Verifies environment variable override works
- External control of WandB behavior

⚠️ **Test 4: Geomancer integration** - Needs testing from Geomancer environment
- Requires manyAgents to be installed
- Should be tested from Geomancer's venv

## How Geomancer Uses This

The manyAgents adapter in Geomancer already sets the correct flags:

```python
# From manyagents/adapters/manylatents_adapter.py:282-292
if logging_mode == 'collect_only':
    # Expert workflow: Disable ALL WandB in manyLatents
    log.info("🔇 DISABLING all WandB in manyLatents (Geomancer will handle logging)")
    overrides['debug'] = True  # ✅ Now works correctly!
    overrides['logger'] = None  # ✅ Now works correctly!
    overrides['callbacks'] = {}  # No callbacks at all
```

With our fixes, manyLatents now properly respects these settings at ALL levels:
1. Top-level WandB initialization
2. PyTorch Lightning trainer loggers
3. Pipeline WandB initialization

## Benefits

1. **No WandB Conflicts**: Multiple manyLatents subprocesses can run concurrently without WandB conflicts
2. **Clean Metrics Return**: Metrics are computed and returned via the EmbeddingOutputs protocol
3. **Centralized Logic**: All WandB disabling logic in one place, easy to maintain
4. **Better Debugging**: Clear log messages indicate when and why WandB is disabled
5. **Multiple Disable Methods**: Supports `logger=None`, `debug=True`, and `WANDB_MODE` env var

## Next Steps for Geomancer

The fixes to manyLatents are complete. Geomancer can now:

1. **Launch manyLatents subprocesses** with `logger=None` or `debug=True`
2. **Collect metrics** from the returned EmbeddingOutputs without WandB interference
3. **Log to its own WandB run** with per-step metrics
4. **Create visualizations** from returned embeddings after all steps complete

The workflow logging infrastructure in `geomancy/utils/logging.py` is already well-designed for this purpose.

## Migration Guide

### For Direct API Users

No changes needed! The API behavior is unchanged for normal usage.

### For Orchestrators (like Geomancer)

Simply set one of these to disable WandB:
```python
from manylatents.api import run

# Option 1: Set logger=None
result = run(data='swissroll', algorithms={...}, logger=None)

# Option 2: Set debug=True
result = run(data='swissroll', algorithms={...}, debug=True)

# Option 3: Set environment variable before import
os.environ['WANDB_MODE'] = 'disabled'
result = run(data='swissroll', algorithms={...})
```

All three methods now consistently disable WandB at all levels.
