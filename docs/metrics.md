# Metrics

The evaluation system for manyLatents: metrics for measuring embedding quality, dataset properties, and algorithm internals. All metric configs live in a flat `configs/metrics/` directory. Each config declares its evaluation target via the `at` field.

## Metric smoke failures: absent parameters and incompatible affinities

`metrics=dse_knn` inherits `k` from the nullable `neighborhood_size` setting.
An absent size (`null` / `None`) resolves to DSE's existing default, **15**, in
the metric itself. Explicit values still reach `compute_knn` unchanged: zero,
negative, noninteger, or `k >= n_samples` requests raise `MeasurementUnavailable`.
The default is not clipped to fit smaller datasets either.

`metrics=mismatch_ratio` with `algorithms/latent=pca` correctly refuses the
pairing. `PCAModule.affinity()` returns
`X_centered @ X_centered.T / (n_samples - 1)`: a scaled sample Gram matrix,
documented by PCA as a covariance, **not a transition matrix**. On the smoke
dataset (5 distributions × 20 points, rotated to 50 dimensions), direct
execution reproduced the negative-weight refusal. Inspection of the fitted
module confirmed exact equality with that formula, negative entries, and row
sums approximately zero. Removing the diagonal, as mismatch does, retains the
negative entries; it cannot turn this matrix into neighborhood probabilities.

Mismatch measures `k_eff = (sum w)^2 / sum(w^2)` for each row. This equals
`1 / sum(p^2)` after normalizing nonnegative weights into probabilities, so
positive row scaling is harmless and rows need not already sum to one (the
diagonal is removed). Signed covariance entries cannot supply that probability
interpretation. The finite, nonnegative, nonempty-row checks remain intact;
substituting a neighborhood size when they fail would invent a result.

The CI metric sweep therefore pairs **only mismatch_ratio with PHATE**, whose
affinity is nonnegative and row-stochastic, instead of PCA. The metric config
still runs; it is not skipped or given a fabricated affinity. Regression tests
in `tests/test_metric_smoke.py` execute both actual pairings through `run()` and
the shared experiment/evaluation engine, assert PCA's refusal and matrix
properties, and compare PHATE's measured `k_eff` against its row probabilities.
They also execute the DSE config's five diffusion times with an absent and an
explicit neighborhood size. Config composition alone did not catch either bug.

## Pipeline Execution Model

Metrics and sampling operate on **named pipeline outputs** — a dict built as `run_experiment()` progresses. Understanding when each output becomes available is key to understanding what `at` and `sampling` can target.

```
run_experiment()
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
│   outputs.update(collect_outputs(algorithm))            ← extras available
│     ├─ registry (manylatents/outputs.py): trajectories,   (algorithm-dependent)
│     │    affinity, adjacency, kernel — off ANY algorithm
│     └─ algorithm.extra_outputs(): its own artifacts only
│
├─ evaluate()   [evaluate.py]
│   ├─ [sampling.embedding] ── subsample before metrics   ← POSITION 2
│   ├─ prewarm_cache() ── kNN/eigenvalues per "on" value
│   └─ for each metric:
│       ├─ read `at` field → resolve from outputs dict
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
| `affinity` | `algorithm.fit()` | output registry → `algorithm.affinity()` | No — algorithm-dependent |
| `kernel` | `algorithm.fit()` | output registry → `algorithm.kernel()` | No — algorithm-dependent |
| `adjacency` | `algorithm.fit()` | output registry → `algorithm.adjacency()` | No — algorithm-dependent |
| `trajectories` | `algorithm.fit()` | output registry → `algorithm.trajectories` | No — algorithm-dependent |

Two ways to add an output, and the first is the one to reach for:

- **Generic** — register an extractor with `@register_output` in `manylatents/outputs.py`. It then runs against *any* algorithm that has the hook, whatever the algorithm inherits. The four above are registered this way; `list_outputs()` enumerates them without fitting anything.
- **Algorithm-specific** — return it from that algorithm's `extra_outputs()` (PCA's robust decomposition, Reeb's node coordinates, Cflows' GRN head). Reserved for real per-algorithm computation, not attribute reads.

`collect_outputs()` merges both, and metrics can immediately target either via `at: "<key>"` — no code changes needed in the evaluation pipeline.

The split exists because generic behaviour used to live on `LatentModule.extra_outputs()`, which meant it reached exactly the classes that inherited it: MIOFlow is a LightningModule, computed `.trajectories`, and could never emit them (#295).

### Sampling positions

Sampling has two categories with different infrastructure:

- **Pre-fit** (`sampling.dataset`): Fixed integration point in `run_experiment()` BEFORE `fit()`. Reduces what the algorithm sees. This is inherently positional — it changes the algorithm's input, not just what metrics evaluate on.
- **Post-fit** (any other key): Dynamic loop in `evaluate()` over the `outputs` dict. Any array-valued output can be sampled. If `sampling.embedding` is configured, the dataset is auto-sliced to matching indices for cross-space metrics.

Post-fit sampling uses the same dynamic resolution as metric routing — it iterates the sampling config, matches keys against the `outputs` dict, and applies the sampler to any matching array. New outputs are automatically sampleable, whether they come from the registry or from `extra_outputs()`.

The `get_indices()` method on samplers accepts `**kwargs` for future extensibility — complex samplers (e.g., diffusion condensation) may need access to the kNN cache, outputs dict, or fitted module to build their sampling operator.

## Metric Selection

Select metrics on the CLI with `metrics=<name>`:

```bash
# Single metric
manylatents algorithms/latent=pca data=swissroll metrics=trustworthiness

# Bundle (composes multiple metrics)
manylatents algorithms/latent=pca data=swissroll metrics=standard
```

## Embedding Metrics

Evaluate the **quality of low-dimensional embeddings**. Compare high-dimensional input to low-dimensional output. Config: `at: embedding`.

{{ metrics_table("embedding") }}

## Module Metrics

Evaluate **algorithm-specific internal components**. Require a fitted module exposing `affinity()` or `kernel()`. Config: `at: module`.

{{ metrics_table("module") }}

## Dataset Metrics

Evaluate properties of the **original high-dimensional data**, independent of the DR algorithm. Config: `at: dataset`.

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
      at: embedding
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

    Set the `at` field in your config to target a pipeline output (see Pipeline Execution Model above):

    - Only needs original data? → `at: dataset`
    - Compares original vs. reduced? → `at: embedding`
    - Needs algorithm internals (affinity, spectral properties)? → `at: module`
    - Needs a specific matrix? → `at: affinity` / `at: kernel` / `at: adjacency` (algorithm must produce it)

    ### Config

    ```yaml
    # configs/metrics/your_metric.yaml
    your_metric:
      _target_: manylatents.metrics.your_metric.YourMetric
      _partial_: true
      k: 10
      at: embedding
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
