# Metrics

The evaluation system for manyLatents: metrics for measuring embedding quality, dataset properties, and algorithm internals. All metric configs live in a flat `configs/metrics/` directory. Each config declares its evaluation target via the `"on"` field.

## Pipeline Execution Model

Metrics and sampling operate on **named pipeline outputs** — a dict built as `run_algorithm()` progresses. Understanding when each output becomes available is key to understanding what `"on"` and `sampling` can target.

```
run_algorithm()
│
├─ datamodule.setup()
│   outputs["dataset"] = ds.data                          ← dataset available
│
├─ [sampling.dataset] ── subsample input before fit       ← POSITION 1
│
├─ algorithm.fit(train_tensor)
├─ algorithm.transform(test_tensor) → embeddings
│   outputs["embedding"] = embeddings                     ← embedding available
│   outputs["module"]    = algorithm                      ← module available
│   outputs["affinity"]  = algorithm.extra_outputs()      ← extras available
│   outputs["kernel"]    = ...                              (algorithm-dependent)
│   outputs["adjacency"] = ...
│
├─ evaluate_outputs()
│   ├─ [sampling.embedding] ── subsample before metrics   ← POSITION 2
│   ├─ prewarm_cache() ── kNN/eigenvalues per "on" value
│   └─ for each metric:
│       ├─ read "on" field → resolve from outputs dict
│       └─ metric_fn(embeddings=..., dataset=..., module=..., cache=...)
│
└─ callbacks (receive full unsampled data + scores)
```

### Output availability

| Output | Available after | Source | Always present |
|---|---|---|---|
| `dataset` | `datamodule.setup()` | `ds.data` | Yes |
| `embedding` | `algorithm.transform()` | Embedding array | Yes |
| `module` | `algorithm.fit()` | Fitted LatentModule | Yes (for LatentModules) |
| `affinity` | `algorithm.fit()` | `module.extra_outputs()` | No — algorithm-dependent |
| `kernel` | `algorithm.fit()` | `module.extra_outputs()` | No — algorithm-dependent |
| `adjacency` | `algorithm.fit()` | `module.extra_outputs()` | No — algorithm-dependent |

New outputs can be added by having a LatentModule return them from `extra_outputs()`. Metrics can immediately target them via `"on": "<key>"` — no code changes needed in the evaluation pipeline.

### Sampling positions

Sampling is keyed by output name but has **two fixed integration points**, not a dynamic loop:

- **`sampling.dataset`** — runs in `run_algorithm()` BEFORE `fit()`. Reduces what the algorithm sees. Embeddings are only produced for the sampled points.
- **`sampling.embedding`** — runs in `evaluate_outputs()` BEFORE metric evaluation. The algorithm sees all data, but metrics evaluate on a subset. Dataset is auto-sliced to matching indices for cross-space metrics.

Other keys (e.g., `sampling.affinity`) are not currently wired. Adding a new sampling position requires adding an integration point in the pipeline code.

### Metric routing vs sampling: what's dynamic, what's fixed

| | Metrics (`"on"` field) | Sampling (config keys) |
|---|---|---|
| **Routing** | Fully dynamic — any key in outputs dict | Two fixed positions (`dataset`, `embedding`) |
| **Adding new targets** | Just return it from `extra_outputs()` | Requires new code in pipeline |
| **Shared vocabulary** | Yes — output names | Yes — same output names |

## Metric Selection

Select metrics on the CLI with `metrics=<name>`:

```bash
# Single metric
manylatents algorithms/latent=pca data=swissroll metrics=trustworthiness

# Bundle (composes multiple metrics)
manylatents algorithms/latent=pca data=swissroll metrics=standard
```

## Embedding Metrics

Evaluate the **quality of low-dimensional embeddings**. Compare high-dimensional input to low-dimensional output. Config: `"on": embedding`.

{{ metrics_table("embedding") }}

## Module Metrics

Evaluate **algorithm-specific internal components**. Require a fitted module exposing `affinity()` or `kernel()`. Config: `"on": module`.

{{ metrics_table("module") }}

## Dataset Metrics

Evaluate properties of the **original high-dimensional data**, independent of the DR algorithm. Config: `"on": dataset`.

{{ metrics_table("dataset") }}

---

=== "Protocol"

    ## Metric Protocol

    All metrics must match the `Metric` protocol (`manylatents/metrics/metric.py`):

    ```python
    def __call__(
        self,
        embeddings: np.ndarray,
        dataset=None,
        module=None,
        cache=None,
    ) -> float | tuple[float, np.ndarray] | dict[str, Any]
    ```

    ### Return Types

    | Type | Use Case | Example |
    |------|----------|---------|
    | `float` | Simple scalar | Trustworthiness: `0.95` |
    | `tuple[float, ndarray]` | Scalar + per-sample | Continuity with `return_per_sample=True` |
    | `dict[str, Any]` | Structured output | Persistent homology: `{'beta_0': ..., 'beta_1': ...}` |

    ## Configuration

    Metrics use Hydra's `_partial_: True` for deferred parameter binding:

    ```yaml
    # configs/metrics/trustworthiness.yaml
    trustworthiness:
      _target_: manylatents.metrics.trustworthiness.Trustworthiness
      _partial_: true
      n_neighbors: 5
      on: embedding
    ```

    ### Multi-Scale Expansion

    List-valued parameters expand via Cartesian product through `flatten_and_unroll_metrics()`:

    ```yaml
    n_neighbors: [5, 10, 20]  # Produces 3 separate evaluations
    ```

    Naming convention: `trustworthiness__n_neighbors_5`, `trustworthiness__n_neighbors_10`, etc.

    ### Shared kNN Cache

    Metrics that need kNN graphs share a cache computed once with `max(k)` across all metrics, avoiding redundant computation.

=== "Writing a New Metric"

    ## Writing a New Metric

    ```python
    import numpy as np
    from typing import Optional

    def YourMetric(
        embeddings: np.ndarray,
        dataset=None,
        module=None,
        k: int = 10,
        cache=None,
    ) -> float:
        # Your computation
        return score
    ```

    ### Choosing the Right Context

    Set the `"on"` field in your config to target a pipeline output (see Pipeline Execution Model above):

    - Only needs original data? → `"on": dataset`
    - Compares original vs. reduced? → `"on": embedding`
    - Needs algorithm internals (affinity, spectral properties)? → `"on": module`
    - Needs a specific matrix? → `"on": affinity` / `"on": kernel` / `"on": adjacency` (algorithm must produce it)

    ### Config

    ```yaml
    # configs/metrics/your_metric.yaml
    your_metric:
      _target_: manylatents.metrics.your_metric.YourMetric
      _partial_: true
      k: 10
      on: embedding
    ```

    ### Testing

    Use `metrics=noop` to verify integration:

    ```bash
    uv run python -m manylatents.main data=swissroll algorithms/latent=pca metrics=noop
    ```

=== "Running Without Metrics"

    ## Null Metrics Support

    manyLatents supports running experiments without metrics computation — useful for fast debugging, exploratory analysis, or workflows where metrics are computed separately.

    ## Usage

    ### CLI (Default)

    Metrics are null by default. Just don't specify them:

    ```bash
    # No metrics (default)
    uv run python -m manylatents.main data=swissroll algorithms/latent=pca

    # With metrics (explicit opt-in)
    uv run python -m manylatents.main data=swissroll algorithms/latent=pca metrics=noop
    ```

    ### Experiment Configs

    ```yaml
    # configs/experiment/my_experiment.yaml
    # @package _global_
    defaults:
      - override /algorithms/latent: pca
      - override /data: swissroll
      - override /callbacks/embedding: default
      # No metrics override - stays null
    ```

    ### Python API

    ```python
    from manylatents.api import run

    result = run(
        data="swissroll",
        algorithms={'latent': 'pca'},
        metrics=None  # Explicitly disable
    )
    ```

    ## Expected Behavior

    When `metrics=null`:

    - Generates embeddings
    - Saves embeddings to files
    - Creates plots (if callbacks configured)
    - Logs to wandb (if configured)
    - Does NOT compute evaluation metrics
    - Shows warning: "No scores found"

    ## Design: Opt-In by Default

    The base config (`configs/config.yaml`) sets metrics to `null`. Experiment configs opt in:

    ```yaml
    # configs/experiment/single_algorithm.yaml
    defaults:
      - override /metrics: noop  # Opt in for this experiment
    ```

    ## Hydra Limitation

    Hydra CLI does not support `null` as an override value. You **cannot** do `metrics=null` on the command line — Hydra's parser converts `"null"` to Python `None`, which its override validator rejects.

    **Workarounds**:

    - Use experiment configs without metrics specified
    - Use the Python API with `metrics=None` (our code handles this)
    - Use `metrics=null` config files (e.g., the base config already does this)

    The API intercepts `None` values before Hydra sees them and sets them after config composition via `OmegaConf.update()`.

    ## Troubleshooting

    ### "Could not find 'metrics/none'"

    You're trying `metrics=none` as a CLI override. Hydra interprets this as looking for `metrics/none.yaml`.

    **Fix**: Use an experiment config, or the API with `metrics=None`.

    ### Metrics Still Being Computed

    Check that:

    1. Your experiment config doesn't have `- override /metrics: ...` in defaults
    2. You're not passing `metrics=...` on the command line
    3. The final config shows `metrics: null`
