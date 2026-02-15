# Representation Probing Guide

This guide explains how to probe neural network representations during training using the manylatents infrastructure.

## Overview

Representation probing extracts activations from model layers at configurable points during training and computes diffusion operators to track how the representation geometry evolves.

**Key components:**
- `LayerSpec` - Specifies which layer to probe
- `ActivationExtractor` - Hooks into the model to capture activations
- `DiffusionGauge` - Converts activations to diffusion operators
- `ActivationTrajectoryCallback` - Orchestrates probing during Lightning training

## Quick Start

Run the default probing experiment:

```bash
python -m manylatents.main experiment=representation_probe \
    algorithms.lightning.config.model_name_or_path=gpt2 \
    trainer.max_epochs=3 \
    logger=wandb
```

This trains GPT-2 on WikiText-2 while probing the last transformer layer every 500 steps.

## Configuration

### Layer Specification

`LayerSpec` defines where to extract activations:

```yaml
layer_specs:
  - path: "transformer.h[-1]"      # Layer path (supports indexing)
    extraction_point: "output"     # What to capture
    reduce: "mean"                 # How to reduce sequence dimension
```

**Path syntax:**
- `transformer.h[-1]` - Last transformer block
- `transformer.h[0]` - First transformer block
- `encoder.layer.5` - Specific layer by index
- `model.embed_tokens` - Embedding layer

**Extraction points:**
- `output` - Layer output activations
- `input` - Layer input activations

**Reduction methods:**
- `mean` - Average over sequence length → `[batch, hidden_dim]`
- `last` - Last token only → `[batch, hidden_dim]`
- `first` - First token (e.g., CLS) → `[batch, hidden_dim]`
- `none` - Keep full sequence → `[batch, seq_len, hidden_dim]`

### Probe Triggers

Control when probing occurs:

```yaml
trigger:
  every_n_steps: 500        # Probe every N training steps
  every_n_epochs: null      # Or every N epochs
  on_checkpoint: true       # Probe when checkpointing
  on_validation_end: true   # Probe after validation
```

Multiple triggers combine with OR logic.

### Diffusion Gauge

Configure how activations become diffusion operators:

```yaml
gauge:
  knn: 15          # k for k-NN graph
  alpha: 1.0       # Gaussian kernel bandwidth
  symmetric: false # Row-stochastic (false) or symmetric normalization
```

The gauge computes: activations → k-NN graph → Gaussian kernel → diffusion operator

## Multi-Layer Probing

Probe multiple layers simultaneously:

```yaml
callbacks:
  trainer:
    probe:
      layer_specs:
        - path: "transformer.h[0]"
          extraction_point: "output"
          reduce: "mean"
        - path: "transformer.h[5]"
          extraction_point: "output"
          reduce: "mean"
        - path: "transformer.h[-1]"
          extraction_point: "output"
          reduce: "mean"
```

## Probe Dataloader

Probing uses a fixed subset of data for consistent comparisons across training. The `TextDataModule` provides this via `probe_dataloader()`:

```yaml
data:
  probe_n_samples: 512  # Size of probe subset
  seed: 42              # Reproducible subset selection
```

## WandB Logging

Enable trajectory logging to WandB:

```yaml
callbacks:
  trainer:
    probe:
      log_to_wandb: true

logger: wandb
```

This logs:
- Diffusion operator eigenspectra
- Trajectory visualizations (if configured)
- Step-indexed operator snapshots

## Programmatic Access

Access probe results after training:

```python
from manylatents.lightning.callbacks.activation_tracker import ActivationTrajectoryCallback

# After trainer.fit()
callback = trainer.callbacks[0]  # Get probe callback
trajectory = callback.get_trajectory()

for step, diffusion_op in trajectory:
    print(f"Step {step}: operator shape {diffusion_op.shape}")
```

## Custom Probes

Extend `DiffusionGauge` for custom probe computations:

```python
from manylatents.callbacks.diffusion_operator import DiffusionGauge
import numpy as np

class MyGauge(DiffusionGauge):
    def __call__(self, activations: np.ndarray) -> np.ndarray:
        # Custom computation
        P = super().__call__(activations)  # Get diffusion operator
        # Add your analysis...
        return P
```

## SLURM Sweeps

Run probing sweeps on cluster:

```bash
python -m manylatents.main -m \
    experiment=representation_probe \
    hydra/launcher=mila_cluster \
    algorithms.lightning.config.model_name_or_path=gpt2,gpt2-medium \
    callbacks.trainer.probe.gauge.knn=10,15,25
```

See `configs/sweep/representation_probe_convergence.yaml` for a full sweep example.

## Architecture

```
┌─────────────────┐
│  HFTrainerModule │ ← Any HuggingFace model
└────────┬────────┘
         │ forward hooks
         ▼
┌─────────────────┐
│ActivationExtractor│ ← Captures layer outputs
└────────┬────────┘
         │ activations [batch, hidden_dim]
         ▼
┌─────────────────┐
│  DiffusionGauge  │ ← k-NN → kernel → operator
└────────┬────────┘
         │ diffusion operator [n_samples, n_samples]
         ▼
┌─────────────────┐
│ WandbProbeLogger │ ← Logs to W&B
└─────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `lightning/hooks.py` | LayerSpec, ActivationExtractor |
| `lightning/callbacks/activation_tracker.py` | ActivationTrajectoryCallback, ProbeTrigger |
| `callbacks/diffusion_operator.py` | DiffusionGauge, build_diffusion_operator |
| `lightning/callbacks/wandb_probe.py` | WandB logging |
| `configs/callbacks/trainer/probe.yaml` | Default probe config |
| `configs/experiment/representation_probe.yaml` | Full experiment config |
