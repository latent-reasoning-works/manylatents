# Interpretability Experiment Infrastructure

**Date:** 2026-04-04
**Status:** Draft
**Scope:** manylatents core changes to enable two experiments

## Context

Two planned experiments require small infrastructure additions to manylatents:

1. **Negative sample rate experiment** — Run UMAP with `negative_sample_rate in {1, 3, 5, 10}`, k-sweep at each. Tests whether the embedding quality cliff position shifts proportionally with negative sample rate, which would identify attraction-repulsion balance as the causal mechanism behind the threshold.

2. **Pythia hidden state experiment** — Extract intermediate layer representations from Pythia language models, run full diagnostic + k-sweep. Tests whether the cliff appears at v ~ 0.1 in LM representation space, which would reframe the finding as a representation learning result rather than biology-specific.

This spec covers only the manylatents infrastructure changes. Experiment configs, sweep definitions, and analysis scripts belong in mbyl_for_practitioners.

## Part 1: UMAP Negative Sample Rate

### Problem

`UMAPModule` does not expose sklearn UMAP's `negative_sample_rate` parameter. The base class `LatentModule.__init__` silently discards unrecognized `**kwargs`, so passing it via config or API has no effect. TorchDR's UMAP does not support this parameter.

### Changes

**`manylatents/algorithms/latent/umap.py`**

Add `negative_sample_rate: int | None = None` to `UMAPModule.__init__`. Store as `self.negative_sample_rate`.

In `_create_model()`, pass to sklearn UMAP only (TorchDR branch unchanged):

```python
# sklearn branch only — add negative_sample_rate
from umap import UMAP
return UMAP(
    n_components=self.n_components,
    random_state=self.random_state,
    n_neighbors=self.n_neighbors,
    min_dist=self.min_dist,
    metric=self.metric,
    n_epochs=self.n_epochs,
    learning_rate=self.learning_rate,
    **({"negative_sample_rate": self.negative_sample_rate}
       if self.negative_sample_rate is not None else {}),
)
```

When `None`, sklearn uses its default (5) — no behavior change for existing users.

**`manylatents/configs/algorithms/latent/umap.yaml`**

Add one line:

```yaml
negative_sample_rate: null
```

### Experiment design (downstream, not in scope)

2D Hydra multirun grid in mbyl_for_practitioners:

```
negative_sample_rate in {1, 3, 5, 10}
neighborhood_size in {3, 5, 7, 10, 15, 20, 25, 30, 50}
```

36 runs. Metrics: `trustworthiness_k`, `continuity_k`. Submitted to Mila cluster via `cluster=mila resources=gpu`.

### Test

New test in `tests/test_umap.py`: verify `negative_sample_rate=1` produces different embeddings than `negative_sample_rate=10` on the same data with the same seed. Confirms the parameter reaches the sklearn constructor.

## Part 2: Pythia Hidden State Infrastructure

### Problem

The probing pipeline (HFTrainerModule + ActivationExtractor + ActivationTrajectoryCallback) is functional but has two gaps for Pythia:

1. `HFTrainerModule.forward()` does not pass `output_hidden_states=True` to the HF model, so the full hidden state tuple is unavailable.
2. No Pythia-specific configs exist — layer paths differ from GPT-2 (`gpt_neox.layers[i]` vs `transformer.h[i]`).

### Changes

**`manylatents/lightning/hf_trainer.py`**

Add `output_hidden_states: bool = False` to `HFTrainerConfig`.

Update `HFTrainerModule.forward()`:

```python
def forward(self, **inputs) -> CausalLMOutput:
    return self.network(
        **inputs,
        output_hidden_states=self.config.output_hidden_states,
    )
```

**`manylatents/configs/algorithms/lightning/hf_trainer.yaml`**

Add under `config:`:

```yaml
output_hidden_states: false
```

**`manylatents/configs/callbacks/trainer/probe_pythia.yaml`** (new file)

```yaml
probe:
  _target_: manylatents.lightning.callbacks.activation_tracker.ActivationTrajectoryCallback
  layer_specs:
    - _target_: manylatents.lightning.hooks.LayerSpec
      path: "gpt_neox.layers[-1]"
      extraction_point: "output"
      reduce: "mean"
  trigger:
    _target_: manylatents.lightning.callbacks.activation_tracker.ProbeTrigger
    every_n_steps: 500
    on_checkpoint: true
    on_validation_end: true
  gauge:
    _target_: manylatents.callbacks.diffusion_operator.DiffusionGauge
    knn: 15
    alpha: 1.0
    symmetric: false
  log_to_wandb: true
```

Users override `layer_specs` via CLI to target specific layers or add multi-layer extraction.

**`manylatents/configs/experiment/representation_probe_pythia.yaml`** (new file)

```yaml
# @package _global_
name: representation_probe_pythia

defaults:
  - override /data: wikitext
  - override /algorithms/lightning: hf_trainer
  - override /trainer: default
  - override /callbacks/trainer: probe_pythia
  - _self_

seed: 42
project: representation_probe
debug: false

data:
  tokenizer_name: "EleutherAI/pythia-70m"
  max_length: 128
  batch_size: 8
  probe_n_samples: 512

algorithms:
  lightning:
    config:
      model_name_or_path: "/network/weights/pythia/pythia-70m"  # local weights; tokenizer from hub
      output_hidden_states: true
      learning_rate: 2e-5
      weight_decay: 0.01
      warmup_steps: 100

trainer:
  max_epochs: 3
  precision: bf16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  val_check_interval: 0.5
  log_every_n_steps: 50
```

Model size is swappable via CLI:

```bash
algorithms.lightning.config.model_name_or_path=/network/weights/pythia/pythia-410m
data.tokenizer_name=EleutherAI/pythia-410m
```

### Pythia weights location

All models available at `/network/weights/pythia/` on the Mila cluster:

| Model | Path |
|-------|------|
| pythia-14m | `/network/weights/pythia/pythia-14m/` |
| pythia-70m | `/network/weights/pythia/pythia-70m/` |
| pythia-160m | `/network/weights/pythia/pythia-160m/` |
| pythia-410m | `/network/weights/pythia/pythia-410m/` |
| pythia-1b | `/network/weights/pythia/pythia-1b/` |
| pythia-1.4b | `/network/weights/pythia/pythia-1.4b/` |
| pythia-2.8b | `/network/weights/pythia/pythia-2.8b/` |
| pythia-6.9b-deduped | `/network/weights/pythia/pythia-6.9b-deduped/` |
| pythia-12b-deduped | `/network/weights/pythia/pythia-12b-deduped/` |

Deduped variants also available for most sizes.

### Test

Integration test with `/network/weights/pythia/pythia-14m` (smallest checkpoint). Verifies:
- HFTrainerModule loads with `output_hidden_states=True`
- Forward pass output contains `.hidden_states` with expected length (n_layers + 1)
- Probe callback fires and produces activations of expected shape

Guarded with `pytest.importorskip("transformers")` and `pytest.mark.skipif(not Path("/network/weights/pythia/pythia-14m").exists(), reason="Pythia weights not available")` so GitHub CI passes.

## File Change Summary

| Change | File | Type |
|---|---|---|
| Add `negative_sample_rate` param | `algorithms/latent/umap.py` | modify |
| Add `negative_sample_rate: null` | `configs/algorithms/latent/umap.yaml` | modify |
| Add `output_hidden_states` field + forward passthrough | `lightning/hf_trainer.py` | modify |
| Add `output_hidden_states: false` | `configs/algorithms/lightning/hf_trainer.yaml` | modify |
| Pythia probe callback config | `configs/callbacks/trainer/probe_pythia.yaml` | create |
| Pythia experiment config | `configs/experiment/representation_probe_pythia.yaml` | create |
| UMAP negative_sample_rate test | `tests/test_umap.py` | modify |
| Pythia integration test | `tests/test_hf_pythia.py` | create |

## Out of Scope

- Multi-layer extraction wrapper / `MultiLayerExtractor`
- Sweep configs, experiment configs for mbyl_for_practitioners
- Analysis scripts, plotting, post-hoc analysis
- New dependencies
