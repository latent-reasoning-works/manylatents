# Null Metrics Support

## Overview

manyLatents supports running experiments without metrics computation. This is useful for:
- Fast embedding generation during debugging
- Exploratory analysis where evaluation isn't needed yet
- Workflows where metrics are computed separately

## Usage

### Via Experiment Configs (Recommended)

Create an experiment config that doesn't specify metrics:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

name: my_experiment

defaults:
  - override /algorithms/latent: pca
  - override /data: swissroll
  - override /callbacks/embedding: default
  # Note: No metrics override - stays null

algorithms:
  latent:
    n_components: 2
```

Run it:
```bash
python -m manylatents.main experiment=my_experiment
```

### Via Python API

When using the programmatic API, you can pass `metrics=None`:

```python
from manylatents.api import run

# Run without metrics
result = run(
    experiment="single_algorithm",
    metrics=None,  # Explicitly disable metrics
    project="my_project"
)
```

This works because the API handles `None` values specially, bypassing Hydra's override parser.

### Via CLI with Opt-In

Start from a minimal base and add only what you need:

```bash
# No metrics (default)
python -m manylatents.main data=swissroll algorithms/latent=pca

# With metrics (explicit opt-in)
python -m manylatents.main data=swissroll algorithms/latent=pca metrics=test_metric
```

## How It Works

### Default Configuration

The base config (`configs/config.yaml`) sets metrics to `null` by default:

```yaml
defaults:
  - metrics: null  # No metrics by default
  - callbacks: default
  - ...
```

### Experiments Can Override

Individual experiment configs can add metrics:

```yaml
# configs/experiment/single_algorithm.yaml
defaults:
  - override /metrics: test_metric  # Add metrics for this experiment
```

### API Handles None Specially

When you pass `metrics=None` to the Python API, it:

1. Filters out the `None` value before Hydra's parser sees it
2. Composes the config without the metrics override
3. Sets `cfg.metrics = None` directly via OmegaConf after composition

This bypasses a Hydra limitation where override strings like `"metrics=None"` fail because Hydra's YAML parser converts them to Python `NoneType`, which the override validator rejects.

## Expected Behavior

When `metrics=null`, the experiment will:

- ✅ Generate embeddings
- ✅ Save embeddings to files
- ✅ Create plots (if callbacks configured)
- ✅ Log to wandb (if configured)
- ❌ Not compute evaluation metrics
- ⚠️ Show warning: "No scores found"

## Examples

### Example 1: Debugging Without Metrics

```bash
# Quick embedding generation for debugging
python -m manylatents.main experiment=single_algorithm_no_metrics
```

### Example 2: API Usage

```python
from manylatents.api import run

# Fast embedding step without evaluation overhead
result = run(
    data="swissroll",
    algorithms={'latent': 'pca'},
    metrics=None  # Skip metrics computation
)
```

### Example 3: Composing from CLI

```bash
# Start with algorithm and data, add callbacks but not metrics
python -m manylatents.main \
    algorithms/latent=pca \
    data=swissroll \
    callbacks/embedding=default
    # metrics stays null (default)
```

## Design Philosophy: Opt-In by Default

manyLatents follows an "opt-in" philosophy for optional features:

1. **Base config is minimal** - Only essential components enabled
2. **Users add features explicitly** - Metrics, callbacks, loggers are opt-in
3. **Experiments provide convenience** - Pre-configured combinations for common use cases
4. **Easy to customize** - Override any part via CLI or API

This design:
- Makes the common case simple
- Avoids fighting Hydra's config system
- Keeps configs composable and maintainable

## Troubleshooting

### "Could not find 'metrics/none'" Error

If you see this error, you're trying to use `metrics=none` as a CLI override. This doesn't work because Hydra interprets it as looking for a config file `metrics/none.yaml`.

**Solution**: Use an experiment config without metrics, or use the API with `metrics=None`.

### "Config group override must be a string or a list. Got NoneType"

This error happens when Hydra's override parser receives a Python `None` value. This is a Hydra limitation.

**Solution**:
- For API: Pass `metrics=None` (our code handles it)
- For CLI: Use experiment configs without metrics specified

### Metrics Still Being Computed

Check that:
1. Your experiment config doesn't have `- override /metrics: ...` in defaults
2. You're not passing `metrics=...` on the command line
3. The config shows `metrics: null` in the final output

## Technical Details

### Why Can't CLI Use `metrics=null`?

Hydra's override system has an architectural constraint:

1. CLI arguments are parsed as YAML strings
2. `"null"` gets converted to Python `None`
3. Override validator checks `isinstance(value, (str, list))`
4. `None` fails this check → error

Our API workaround:
- Intercepts overrides before Hydra sees them
- Filters out `None` values
- Sets them after config composition via `OmegaConf.update()`

### Code References

**API fix**: `manylatents/api.py` lines 94-122
```python
# Filter None values before Hydra override parsing
none_keys = []
for key, value in overrides.items():
    if value is None:
        none_keys.append(key)
        continue
    # ... handle other types

# Set None values after composition
for key in none_keys:
    OmegaConf.update(cfg, key, None, merge=False)
```

**Experiment evaluation**: `manylatents/experiment.py` lines 137, 146, 184
```python
# Code already handles None gracefully
metric_cfgs = flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else {}
```

## Related Documentation

- [Config System Overview](./config_system.md)
- [API Reference](./api_reference.md)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
