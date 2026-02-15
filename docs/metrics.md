# Metrics

The evaluation system for manyLatents: a three-level architecture for measuring embedding quality, dataset properties, and algorithm internals.

## Embedding Metrics

Evaluate the **quality of low-dimensional embeddings**. Compare high-dimensional input to low-dimensional output.

{{ metrics_table("embedding") }}

Config pattern: `metrics/embedding=<name>`

## Module Metrics

Evaluate **algorithm-specific internal components**. Require a fitted module exposing `affinity_matrix()` or `kernel_matrix()`.

{{ metrics_table("module") }}

Config pattern: `metrics/module=<name>`

## Dataset Metrics

Evaluate properties of the **original high-dimensional data**, independent of the DR algorithm.

{{ metrics_table("dataset") }}

Config pattern: `metrics/dataset=<name>`

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
    # configs/metrics/embedding/trustworthiness.yaml
    _target_: manylatents.metrics.trustworthiness.Trustworthiness
    _partial_: true
    n_neighbors: 5
    ```

    ### Multi-Scale Expansion

    List-valued parameters expand via Cartesian product through `flatten_and_unroll_metrics()`:

    ```yaml
    n_neighbors: [5, 10, 20]  # Produces 3 separate evaluations
    ```

    Naming convention: `embedding.trustworthiness__n_neighbors_5`, `embedding.trustworthiness__n_neighbors_10`, etc.

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

    ### Choosing the Right Level

    - Only needs original data? → `metrics/dataset/`
    - Compares original vs. reduced? → `metrics/embedding/`
    - Needs algorithm internals? → `metrics/module/`

    ### Config

    ```yaml
    # configs/metrics/embedding/your_metric.yaml
    _target_: manylatents.metrics.your_metric.YourMetric
    _partial_: true
    k: 10
    ```

    ### Testing

    Use `metrics=test_metric` to verify integration:

    ```bash
    python -m manylatents.main data=swissroll algorithms/latent=pca metrics=test_metric
    ```

=== "Running Without Metrics"

    ## Null Metrics Support

    manyLatents supports running experiments without metrics computation — useful for fast debugging, exploratory analysis, or workflows where metrics are computed separately.

    ## Usage

    ### CLI (Default)

    Metrics are null by default. Just don't specify them:

    ```bash
    # No metrics (default)
    python -m manylatents.main data=swissroll algorithms/latent=pca

    # With metrics (explicit opt-in)
    python -m manylatents.main data=swissroll algorithms/latent=pca metrics=test_metric
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
      - override /metrics: test_metric  # Opt in for this experiment
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
