# Probing Architecture

**Status:** Accepted
**Date:** 2026-02-02

## Context

We needed to add representation probing capabilities to extract and analyze intermediate representations during training (Lightning) or after inference (LatentModule). Initial implementations created a separate `manylatents/probing/` module with `DiffusionGauge`, `TrajectoryVisualizer`, etc.

The question arose: should probing be its own top-level module, or integrated into the existing callback system?

## Decision

**Probing is part of the callback system, not a separate module.**

Core probing utilities live in `manylatents/callbacks/probing.py`. Lightning-specific wrappers live in `manylatents/lightning/callbacks/probing.py`.

## Rationale

### Probes are observers, not pipeline stages

Probes hook into the training/inference process to observe and compute derived quantities. They don't transform the main data flow like a `LatentModule` would. This matches the callback mental model:

```
LatentModule: data → fit() → transform() → embeddings  (pipeline stage)
Probe:        embeddings → observe → derived_output    (hook/observer)
```

### Avoid proliferating top-level modules

The codebase already has:
- `metrics/` - scalar measurements from embeddings
- `callbacks/` - post-hoc operations on outputs
- `algorithms/` - transformations (LatentModule, LightningModule)

Adding `probing/` as a fourth measurement/observation concept fragments the architecture. Probes fit naturally as callbacks that compute non-scalar derived quantities.

### Unified dispatch pattern

Like `evaluate()` dispatches based on input type, `probe()` dispatches based on input:

```python
from manylatents.callbacks.probing import probe

# Works with both LatentModule outputs and Lightning activations
diff_op = probe(embeddings, method="diffusion")
diff_op = probe(activations, method="diffusion")
```

### Future extensibility

New probe types (SAE features, attention patterns, etc.) are added as methods:

```python
@probe.register(Tensor)
def _probe_tensor(source, /, method="diffusion", **kwargs):
    if method == "diffusion":
        return DiffusionGauge(**kwargs)(source)
    elif method == "sae":
        return SAEProbe(**kwargs)(source)
    elif method == "attention":
        return AttentionProbe(**kwargs)(source)
```

No new modules needed - just new probe implementations within the callback.

## Structure

```
manylatents/
  callbacks/
    probing.py              # Core: probe(), DiffusionGauge, TrajectoryVisualizer
    tests/test_probing.py
    embedding/              # Existing embedding callbacks

  lightning/callbacks/
    probing.py              # RepresentationProbeCallback (Lightning-specific)
    wandb_probe.py          # WandB logging for probe outputs
    tests/test_probing.py
```

## Consequences

### Positive
- No new top-level modules to maintain
- Clear separation: callbacks observe, algorithms transform
- Consistent with existing dispatch patterns (`evaluate()`)
- Easy to add new probe types

### Negative
- `callbacks/probing.py` may grow large with many probe types
- Trajectory visualization (PHATE-based) feels slightly out of place in callbacks

### Mitigations
- Split into `callbacks/probing/` subpackage if it grows too large
- Trajectory analysis is specifically for analyzing probe outputs over time, so it belongs with probing
