# HuggingFace Representation Audit Architecture

## Overview

Infrastructure for auditing neural network representations during training, computing diffusion operators from activations, and merging representations across models.

## Process Model

**Same process, PyTorch forward hooks, Lightning callback orchestration**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  trainer.fit(hf_module, datamodule)  ← Single Python process            │
│                                                                          │
│    Training Loop                                                         │
│    ┌──────────────────────────────────────────────────────────────────┐ │
│    │ for batch in train_loader:                                        │ │
│    │     loss = hf_module.training_step(batch)                         │ │
│    │     loss.backward()                                               │ │
│    │     optimizer.step()                                              │ │
│    │                                                                   │ │
│    │     # Lightning fires callback                                    │ │
│    │     callback.on_train_batch_end(trainer, hf_module, ...)          │ │
│    │              │                                                    │ │
│    │              ▼                                                    │ │
│    │     ┌────────────────────────────────────────────────────────┐   │ │
│    │     │  RepresentationAuditCallback                           │   │ │
│    │     │                                                        │   │ │
│    │     │  if step % audit_every == 0:                           │   │ │
│    │     │      model = hf_module.network  # SAME OBJECT IN RAM   │   │ │
│    │     │                                                        │   │ │
│    │     │      # Register hooks on specified layers              │   │ │
│    │     │      for spec in layer_specs:                          │   │ │
│    │     │          layer = resolve(model, spec)  # e.g., [-1]    │   │ │
│    │     │          layer.register_forward_hook(capture_fn)       │   │ │
│    │     │                                                        │   │ │
│    │     │      # Forward probe set through model                 │   │ │
│    │     │      with torch.no_grad():                             │   │ │
│    │     │          for probe_batch in probe_loader:              │   │ │
│    │     │              model(**probe_batch)  # hooks fire        │   │ │
│    │     │                                                        │   │ │
│    │     │      # activations dict now populated                  │   │ │
│    │     │      diff_op = DiffusionGauge(activations["layer-1"])  │   │ │
│    │     │      self.trajectory.append((step, diff_op))           │   │ │
│    │     └────────────────────────────────────────────────────────┘   │ │
│    └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Insight

**Not a chained API** - the HF model is a Python object in memory. We access it directly via `hf_module.network.model.layers[-1]` and register PyTorch forward hooks.

## Component Breakdown

| Component | Location | Purpose |
|-----------|----------|---------|
| `HFTrainerModule` | `manylatents/lightning/hf_trainer.py` | LightningModule wrapping AutoModel* |
| `ActivationExtractor` | `manylatents/lightning/hooks.py` | Register hooks, collect activations by LayerSpec |
| `DiffusionGauge` | `manylatents/gauge/diffusion.py` | activations → kernel → diffusion operator |
| `RepresentationAuditCallback` | `manylatents/lightning/callbacks.py` | Orchestrates extraction at triggers |
| `DiffusionMerging` | `manylatents/algorithms/merging.py` | Extend MergingModule for operator fusion |

## Layer Specification

Flexible layer targeting:

```python
layer_specs = [
    "model.layers[-1]",           # Last transformer block
    "model.layers[-1].self_attn", # Attention in last block
    "model.layers[12]",           # Specific layer by index
    "model.lm_head",              # Output projection
]
```

Or structured:

```python
@dataclass
class LayerSpec:
    path: str                         # "model.layers[-1]"
    extraction_point: str = "output"  # "output", "input", "hidden_states"
    reduce: str = "mean"              # "mean", "last_token", "cls", "all"
```

## Diffusion Operator Merging Strategies

| Strategy | Formula | Notes |
|----------|---------|-------|
| Weighted interpolation | `P* = Σ w_i P_i` | Normalize to stochastic after |
| Frobenius mean | `P* = (1/N) Σ P_i` | Closed-form solution to `argmin_P Σ \|\|P - P_i\|\|_F²` |
| OT barycenter | Wasserstein barycenter | Via POT library, added later |

## Target Experiments

### Merging Experiment
1. Select pretrained LLMs (multiple sizes/recipes)
2. Fix probe dataset D_probe
3. Extract penultimate-layer activations on D_probe
4. Compute pairwise distance → affinity (Gaussian kernel)
5. Construct diffusion operator P_i per model
6. Merge operators → P* (interpolation or OT barycenter)
7. PHATE embedding from P*
8. Fine-tune student with joint objective matching target embedding

### Convergent Training Trajectories
1. Train N LLM replicas (varying seed, scale, hyperparams)
2. Fix probe set, record activations at checkpoints t
3. Construct P_{i,t} per model per time
4. PHATE embed {P_{i,t}} to visualize trajectories
5. Quantify convergence via spread metric over t

## Trigger Configuration

```python
@dataclass
class AuditTrigger:
    every_n_steps: int | None = None      # Step-based
    every_n_epochs: int | None = None     # Epoch-based
    on_checkpoint: bool = False           # When checkpoints saved
    on_validation_end: bool = False       # After validation
```
