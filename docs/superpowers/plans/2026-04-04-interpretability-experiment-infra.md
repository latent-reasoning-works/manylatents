# Interpretability Experiment Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose UMAP `negative_sample_rate` for causal mechanism experiments and wire Pythia hidden-state probing infrastructure.

**Architecture:** Two independent changes. (1) Add one parameter to `UMAPModule`, pass it to sklearn UMAP. (2) Add `output_hidden_states` to `HFTrainerConfig`, create Pythia-specific probe and experiment configs.

**Tech Stack:** umap-learn, transformers (HF), PyTorch Lightning, Hydra

**Spec:** `docs/superpowers/specs/2026-04-04-interpretability-experiment-infra-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `manylatents/algorithms/latent/umap.py` | Add `negative_sample_rate` param, pass to sklearn |
| Modify | `manylatents/configs/algorithms/latent/umap.yaml` | Expose param in config |
| Modify | `tests/test_umap_backend.py` | Test negative_sample_rate reaches sklearn |
| Modify | `manylatents/lightning/hf_trainer.py` | Add `output_hidden_states` to config + forward |
| Modify | `manylatents/configs/algorithms/lightning/hf_trainer.yaml` | Expose param in config |
| Create | `manylatents/configs/callbacks/trainer/probe_pythia.yaml` | Pythia layer paths for probe callback |
| Create | `manylatents/configs/experiment/representation_probe_pythia.yaml` | End-to-end Pythia probe experiment |
| Modify | `manylatents/lightning/tests/test_hf_trainer.py` | Test output_hidden_states passthrough |

---

### Task 1: UMAP `negative_sample_rate` — Test + Implementation

**Files:**
- Modify: `tests/test_umap_backend.py`
- Modify: `manylatents/algorithms/latent/umap.py:13-67`

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_umap_backend.py`:

```python
def test_umap_negative_sample_rate_affects_embedding():
    """negative_sample_rate should reach sklearn UMAP and change the embedding."""
    from manylatents.algorithms.latent.umap import UMAPModule

    x = torch.randn(80, 10, generator=torch.Generator().manual_seed(0))

    m1 = UMAPModule(
        n_components=2, random_state=42, n_neighbors=10,
        n_epochs=50, negative_sample_rate=1,
    )
    emb1 = m1.fit_transform(x)

    m2 = UMAPModule(
        n_components=2, random_state=42, n_neighbors=10,
        n_epochs=50, negative_sample_rate=10,
    )
    emb2 = m2.fit_transform(x)

    # Same seed, same data, different negative_sample_rate -> different embeddings
    assert emb1.shape == emb2.shape == (80, 2)
    assert not np.allclose(emb1, emb2, atol=1e-3), (
        "Embeddings should differ when negative_sample_rate changes"
    )


def test_umap_negative_sample_rate_none_uses_default():
    """negative_sample_rate=None should not change default sklearn behavior."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, n_neighbors=5, n_epochs=10)
    assert m.negative_sample_rate is None
    # sklearn default is 5 — verify it's not overridden
    from umap import UMAP as SklearnUMAP
    assert m.model.negative_sample_rate == SklearnUMAP().negative_sample_rate
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_umap_backend.py::test_umap_negative_sample_rate_affects_embedding -v`

Expected: FAIL — `TypeError: UMAPModule.__init__() got an unexpected keyword argument 'negative_sample_rate'`

- [ ] **Step 3: Add `negative_sample_rate` to UMAPModule**

In `manylatents/algorithms/latent/umap.py`, add the parameter to `__init__` and pass it through in `_create_model()`.

In `__init__`, add `negative_sample_rate: int | None = None` after `fit_fraction` (line 22), and store it:

Replace the `__init__` parameter list and body (lines 14–43):

```python
class UMAPModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = 'euclidean',
        n_epochs: Optional[int] = 200,
        learning_rate: float = 1.0,
        fit_fraction: float = 1.0,
        negative_sample_rate: int | None = None,
        backend: str | None = None,
        device: str | None = None,
        neighborhood_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device,
            neighborhood_size=neighborhood_size, **kwargs,
        )
        self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fit_fraction = fit_fraction
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state

        self._resolved_backend = resolve_backend(backend)
        self.model = self._create_model()
```

In `_create_model()`, pass `negative_sample_rate` to the sklearn branch only. Replace the else branch (lines 56–67):

```python
    def _create_model(self):
        if self._resolved_backend == "torchdr":
            from torchdr import UMAP

            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                device=resolve_device(self.device),
                random_state=self.random_state,
            )
        else:
            from umap import UMAP

            kwargs = {}
            if self.negative_sample_rate is not None:
                kwargs["negative_sample_rate"] = self.negative_sample_rate
            return UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
                **kwargs,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_umap_backend.py -v`

Expected: ALL PASS (including existing tests — no regressions)

- [ ] **Step 5: Commit**

```bash
git add tests/test_umap_backend.py manylatents/algorithms/latent/umap.py
git commit -m "feat: expose negative_sample_rate in UMAPModule for sklearn backend"
```

---

### Task 2: UMAP `negative_sample_rate` — Config

**Files:**
- Modify: `manylatents/configs/algorithms/latent/umap.yaml`

- [ ] **Step 1: Add the parameter to the Hydra config**

Add `negative_sample_rate: null` to `manylatents/configs/algorithms/latent/umap.yaml` after `fit_fraction`:

```yaml
_target_: manylatents.algorithms.latent.umap.UMAPModule
n_components: 2
random_state: ${seed}
n_neighbors: 15
min_dist: 0.5
n_epochs: 500
metric: 'euclidean'
learning_rate: 1.0
fit_fraction: 1.0
negative_sample_rate: null
neighborhood_size: ${neighborhood_size}
backend: null
device: null
```

- [ ] **Step 2: Verify Hydra config composes correctly**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hydra_configs.py -v -k "umap"`

Expected: PASS — the config group test should still compose without errors. If no Hydra config test exists for umap specifically, run the full config test suite:

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hydra_configs.py -v`

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add manylatents/configs/algorithms/latent/umap.yaml
git commit -m "config: add negative_sample_rate to umap.yaml (default null)"
```

---

### Task 3: HFTrainerModule `output_hidden_states` — Test + Implementation

**Files:**
- Modify: `manylatents/lightning/tests/test_hf_trainer.py`
- Modify: `manylatents/lightning/hf_trainer.py:17-85`

- [ ] **Step 1: Write the failing test**

Add to the end of `manylatents/lightning/tests/test_hf_trainer.py`:

```python
def test_hf_trainer_config_output_hidden_states_default():
    """output_hidden_states should default to False."""
    config = HFTrainerConfig(model_name_or_path="gpt2")
    assert config.output_hidden_states is False


@pytest.mark.slow
def test_hf_trainer_output_hidden_states():
    """forward() with output_hidden_states=True returns hidden state tuple."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        output_hidden_states=True,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world", "Test input"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )

    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.hidden_states is not None
    # hidden_states is a tuple of (n_layers + 1) tensors (embedding + each layer)
    n_layers = module.network.config.num_hidden_layers
    assert len(outputs.hidden_states) == n_layers + 1
    # Each tensor: (batch, seq_len, hidden_dim)
    assert outputs.hidden_states[0].shape[0] == 2  # batch_size


@pytest.mark.slow
def test_hf_trainer_output_hidden_states_disabled():
    """forward() with output_hidden_states=False returns None for hidden_states."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        output_hidden_states=False,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )

    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.hidden_states is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/lightning/tests/test_hf_trainer.py::test_hf_trainer_config_output_hidden_states_default -v`

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'output_hidden_states'`

- [ ] **Step 3: Add `output_hidden_states` to HFTrainerConfig and forward()**

In `manylatents/lightning/hf_trainer.py`, add the field to `HFTrainerConfig` (after `attn_implementation`, before `device_map`):

```python
@dataclass
class HFTrainerConfig:
    """Configuration for HFTrainerModule.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps for scheduler
        adam_epsilon: Epsilon for AdamW
        torch_dtype: Optional dtype for model (e.g., torch.bfloat16)
        trust_remote_code: Whether to trust remote code for model loading
        attn_implementation: Attention implementation (e.g., "flash_attention_2")
        output_hidden_states: Whether to return hidden states from all layers
        device_map: Device placement strategy (e.g., "auto"). None lets
            Lightning/FSDP manage placement (existing behavior).
    """
    model_name_or_path: str
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    output_hidden_states: bool = False
    device_map: Optional[str] = None
```

Update `forward()` in `HFTrainerModule`:

```python
    def forward(self, **inputs) -> CausalLMOutput:
        """Forward pass through the model."""
        return self.network(
            **inputs,
            output_hidden_states=self.config.output_hidden_states,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/lightning/tests/test_hf_trainer.py -v`

Expected: ALL PASS (including all existing tests — no regressions)

- [ ] **Step 5: Commit**

```bash
git add manylatents/lightning/hf_trainer.py manylatents/lightning/tests/test_hf_trainer.py
git commit -m "feat: add output_hidden_states to HFTrainerConfig for multi-layer extraction"
```

---

### Task 4: HF Trainer Config YAML + Pythia Probe Config

**Files:**
- Modify: `manylatents/configs/algorithms/lightning/hf_trainer.yaml`
- Create: `manylatents/configs/callbacks/trainer/probe_pythia.yaml`

- [ ] **Step 1: Add `output_hidden_states` to hf_trainer.yaml**

Update `manylatents/configs/algorithms/lightning/hf_trainer.yaml`:

```yaml
# manylatents/configs/algorithms/lightning/hf_trainer.yaml
# HuggingFace causal LM trainer configuration
_target_: manylatents.lightning.hf_trainer.HFTrainerModule

config:
  _target_: manylatents.lightning.hf_trainer.HFTrainerConfig
  model_name_or_path: "gpt2"
  learning_rate: 2e-5
  weight_decay: 0.0
  warmup_steps: 100
  adam_epsilon: 1e-8
  trust_remote_code: false
  output_hidden_states: false
  # Optional: torch_dtype, attn_implementation
```

- [ ] **Step 2: Create Pythia probe callback config**

Create `manylatents/configs/callbacks/trainer/probe_pythia.yaml`:

```yaml
# Pythia representation probe callback configuration
# Layer path uses gpt_neox.layers[] (GPT-NeoX architecture)
# Override layer_specs via CLI for multi-layer or specific-layer extraction
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

- [ ] **Step 3: Verify Hydra config composition**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hydra_configs.py -v`

Expected: ALL PASS — new config files should be picked up by the composition test.

- [ ] **Step 4: Commit**

```bash
git add manylatents/configs/algorithms/lightning/hf_trainer.yaml manylatents/configs/callbacks/trainer/probe_pythia.yaml
git commit -m "config: add output_hidden_states to hf_trainer, add probe_pythia callback"
```

---

### Task 5: Pythia Experiment Config

**Files:**
- Create: `manylatents/configs/experiment/representation_probe_pythia.yaml`

- [ ] **Step 1: Create the experiment config**

Create `manylatents/configs/experiment/representation_probe_pythia.yaml`:

```yaml
# @package _global_
#
# End-to-end experiment for representation probing on Pythia models.
#
# Usage:
#   python -m manylatents.main experiment=representation_probe_pythia
#   python -m manylatents.main experiment=representation_probe_pythia \
#       algorithms.lightning.config.model_name_or_path=/network/weights/pythia/pythia-410m \
#       data.tokenizer_name=EleutherAI/pythia-410m
#
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

# --- Data Configuration ---
data:
  tokenizer_name: "EleutherAI/pythia-70m"
  max_length: 128
  batch_size: 8
  probe_n_samples: 512

# --- Model Configuration ---
# Weights are local on Mila cluster; tokenizer loaded from HuggingFace hub
algorithms:
  lightning:
    config:
      model_name_or_path: "/network/weights/pythia/pythia-70m"
      output_hidden_states: true
      learning_rate: 2e-5
      weight_decay: 0.01
      warmup_steps: 100

# --- Trainer Configuration ---
trainer:
  max_epochs: 3
  precision: bf16-mixed
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4
  val_check_interval: 0.5
  log_every_n_steps: 50
```

- [ ] **Step 2: Verify Hydra config composition**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hydra_configs.py -v -k "representation_probe_pythia or experiment"`

If no experiment-specific Hydra test exists, verify manually:

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run python -c "from hydra import compose, initialize_config_dir; initialize_config_dir(config_dir='manylatents/configs', version_base=None); cfg = compose(config_name='config', overrides=['experiment=representation_probe_pythia']); print(cfg.algorithms.lightning.config.model_name_or_path)"`

Expected: `/network/weights/pythia/pythia-70m`

- [ ] **Step 3: Commit**

```bash
git add manylatents/configs/experiment/representation_probe_pythia.yaml
git commit -m "config: add representation_probe_pythia experiment config"
```

---

### Task 6: Final Verification

**Files:** None (test-only)

- [ ] **Step 1: Run the full test suite to check for regressions**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/ -x -q && uv run pytest manylatents/callbacks/tests/ -x -q`

Expected: ALL PASS

- [ ] **Step 2: Run slow HF tests specifically**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest manylatents/lightning/tests/test_hf_trainer.py -v -m slow`

Expected: ALL PASS (including the new `output_hidden_states` tests)

- [ ] **Step 3: Run the Hydra config composition tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && uv run pytest tests/test_hydra_configs.py -v`

Expected: ALL PASS (all config groups compose, including new `probe_pythia` and `representation_probe_pythia`)
