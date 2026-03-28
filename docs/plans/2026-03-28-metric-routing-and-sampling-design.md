# Metric Routing and Sampling Unification

**Date:** 2026-03-28
**Status:** Implemented
**Issues:** #218 (metric config groups disconnect), #249 (pre-algorithm sampling)

## Problem

Three interconnected problems in `experiment.py`:

1. **Metric config groups don't dispatch.** The `embedding/`, `dataset/`, `module/` directories under `configs/metrics/` imply that metrics are dispatched to different inputs based on their group. In practice, every metric receives the identical `(embeddings, dataset, module, cache)` call. The groups serve only as Hydra composition units and result namespace prefixes.

2. **Sampling is hardwired to one point.** The `SamplingStrategy` system only runs post-algorithm inside `evaluate_outputs()` (experiment.py:344-351). There is no way to subsample input data before `fit()`, which matters for large imbalanced datasets (issue #249).

3. **Cache consistency is enforced by hardcoded frozensets.** `_DATA_KNN_METRICS` and `_SPECTRAL_METRICS` in `experiment.py` hardcode which metrics need which cache entries. Adding a new metric that operates on a non-default space requires editing these frozensets.

These are the same problem: metrics and sampling need to be pointable at any data the pipeline produces, and the cache must stay consistent with whatever data a metric actually evaluates on.

## Design

### Core Concept

Every metric config has a required `on` field that names which pipeline output it evaluates on. Sampling is keyed by the same output names. Cache prewarming is derived from `on` values, not hardcoded lists.

### Pipeline Outputs

`evaluate_outputs()` builds an `outputs` dict as the pipeline progresses:

```python
outputs = {}
outputs["dataset"] = ds.data        # always present
outputs["embedding"] = embeddings   # after transform()

# Dynamic extras from the algorithm (normalized names)
if module is not None:
    outputs["module"] = module
    for key, val in module.extra_outputs().items():
        outputs[key] = val
```

Output keys use short normalized names: `affinity`, `kernel`, `adjacency` (not `affinity_matrix`). The `LatentModule.extra_outputs()` method and its callers are updated to use these short names. This is a protocol-level convention documented on the base class.

### The `on` Field

Every metric config must include `on: <output_name>`. This is the routing directive — it names which pipeline output the metric evaluates on.

```yaml
# configs/metrics/trustworthiness.yaml
trustworthiness:
  _target_: manylatents.metrics.trustworthiness.Trustworthiness
  _partial_: True
  n_neighbors: 5
  on: embedding

# configs/metrics/dse_knn.yaml
dse_knn:
  _target_: manylatents.metrics.diffusion_spectral_entropy.DiffusionSpectralEntropy
  _partial_: True
  on: embedding
  k: ${neighborhood_size}
```

Valid `on` values: `embedding`, `dataset`, `module`, plus any key produced by `module.extra_outputs()` (`affinity`, `kernel`, `adjacency`).

For `on: embedding` and `on: dataset`, the resolved array goes into the metric's `embeddings` kwarg as the primary evaluation data. For `on: module`, the metric receives the fitted module via the `module` kwarg — no array substitution (module-level metrics access internal state like affinity matrices through the module's methods). Cache prewarming adapts accordingly: kNN for array-valued outputs, eigenvalue decomposition for module.

#### Sweeping `on`

`on` participates in `flatten_and_unroll_metrics()` sweep expansion like any other list-valued field:

```yaml
dse_both_spaces:
  _target_: manylatents.metrics.diffusion_spectral_entropy.DiffusionSpectralEntropy
  _partial_: True
  on: [embedding, dataset]
  k: ${neighborhood_size}
```

This produces two evaluations: `dse_both_spaces__on_embedding` and `dse_both_spaces__on_dataset`. Same suffix pattern as all other sweeps (`__key_value`).

### Routing in evaluate_outputs()

For each metric:

1. Read `on` from config (required field — error if missing)
2. Resolve the output:
   - `on: embedding` or `on: dataset` → resolve `outputs[on_value]` to an array, pass as `embeddings` kwarg
   - `on: module` → no array substitution; metric accesses module internals via `module` kwarg
   - `on: affinity`, `on: kernel`, etc. → resolve from `outputs` dict (populated by `module.extra_outputs()`)
3. If the output doesn't exist (e.g., `on: affinity` against PCA), **skip with loud warning**: `logger.warning(f"Skipping metric '{metric_name}': output '{on_value}' not available from {type(module).__name__}. Check that the algorithm produces this output.")`
4. Pop `on` from a config copy before `hydra.utils.instantiate()` so it doesn't leak into metric kwargs
5. Call the metric with the standard signature — `metric_fn(embeddings=..., dataset=..., module=..., cache=...)` — unchanged

The `on` field determines:
- Which data is primary for the metric (array outputs → `embeddings` kwarg; module → `module` kwarg)
- Which cache entries to prewarm (kNN for array outputs, eigenvalues for module)
- Which sampling indices apply

The metric still receives `dataset` and `module` as additional context (for cross-space metrics like trustworthiness). The protocol signature does not change.

### Pre-fit Sampling Integration

Pre-fit sampling (`sampling.dataset`) runs in `run_algorithm()`, NOT in `evaluate_outputs()`. Integration point is after tensor extraction, before `execute_step()`:

```python
# run_algorithm() — after line 616 (torch.cat), before execute_step()
train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
test_tensor  = torch.cat([b[field_index].cpu() for b in test_loader],  dim=0)

# NEW: pre-fit sampling
sampling_cfg = cfg.get("sampling", None)
pre_fit_indices = None
if sampling_cfg is not None and "dataset" in sampling_cfg:
    sampler = hydra.utils.instantiate(sampling_cfg["dataset"])
    pre_fit_indices = sampler.get_indices(train_tensor.numpy())
    train_tensor = train_tensor[pre_fit_indices]
    test_tensor = test_tensor[pre_fit_indices]
    if train_labels is not None:
        train_labels = train_labels[pre_fit_indices]

# execute_step() sees only the sampled data
latents = execute_step(algorithm=algorithm, train_tensor=train_tensor, ...)
```

Post-fit sampling (`sampling.embedding`) remains in `evaluate_outputs()`, applied before metric evaluation.

### Sampling

#### Config Structure

Sampling moves from `configs/metrics/sampling/` to a top-level `configs/sampling/` config group. Keyed by output names — same vocabulary as `on`:

```yaml
# configs/sampling/balanced.yaml
dataset:
  _target_: manylatents.utils.sampling.StratifiedSampling
  fraction: 0.1

# configs/sampling/balanced_with_eval_subsample.yaml
dataset:
  _target_: manylatents.utils.sampling.StratifiedSampling
  fraction: 0.1
embedding:
  _target_: manylatents.utils.sampling.RandomSampling
  fraction: 0.5
```

```yaml
# config.yaml addition
- optional sampling: null
```

#### Index Propagation

Sampling at `dataset` level means the algorithm sees fewer points. Embeddings are produced only for the sampled points. This is causal, not a rule — there are fewer points because the algorithm only received fewer points.

Sampling at `embedding` level is a further subsample of whatever the algorithm produced.

Cross-space metrics (e.g., trustworthiness needs both embedding and dataset arrays) always use the **embedding-level indices** in both spaces. Rationale: reducing the embedding space is the common case (large dataset, evaluate metrics on a subset). Starting from a small dataset and ending up with more embeddings than inputs is rare. The embedding indices are the most restrictive set, so this is effectively "most restrictive wins."

#### Sampler Protocol Extension

Add `get_indices()` to the `SamplingStrategy` protocol for the new routing system:

```python
class SamplingStrategy(Protocol):
    def get_indices(
        self,
        data: np.ndarray,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Return indices for the subsample. Does not slice data."""
        ...

    def sample(
        self,
        embeddings: np.ndarray,
        dataset: object,
        n_samples: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """Legacy interface. Returns (subsampled_embeddings, subsampled_dataset, indices)."""
        ...
```

Existing implementations already have `get_indices()` on `RandomSampling`. Add it to the others.

### Cache Prewarming (No Hardcoding)

Delete `_DATA_KNN_METRICS` and `_SPECTRAL_METRICS` frozensets.

Rewrite `extract_k_requirements()`:

1. For each metric, read its `on` value (required)
2. Extract `k`/`n_neighbors` from the config (existing logic, unchanged)
3. Group k requirements by `on` value — not by hardcoded target string
4. Return `{on_value: set_of_k_values}` for array-valued outputs, plus `spectral: True` if any metric has `on: module`

`prewarm_cache()` iterates the requirements dict:

```python
for on_value, k_set in requirements.items():
    data = outputs.get(on_value)
    if data is not None and k_set:
        max_k = max(k_set)
        compute_knn(data, k=max_k, cache=cache)
```

No special casing by name. If a new output type needs kNN, it gets kNN — because a metric pointed at it and requested a k value.

Eigenvalue prewarming follows the same pattern: if any metric's `on` value resolves to `module` and the module is available, prewarm eigenvalues. No target-string matching — the `on` value alone drives it.

### Output Naming Convention

`LatentModule.extra_outputs()` returns short names:

| Current | Normalized |
|---|---|
| `affinity_matrix` | `affinity` |
| `kernel_matrix` | `kernel` |
| `adjacency_matrix` | `adjacency` |

The base class documents the convention. Subclasses that override `extra_outputs()` use the short names. The methods on LatentModule are renamed accordingly (`affinity()`, `kernel()`, `adjacency()`), with the old names as deprecated aliases.

### Backward Compatibility

This is a **breaking change** to config structure. All metric configs are rewritten.

- **Metric configs:** Moved from 3 subdirectories to flat `configs/metrics/`. Every config gets explicit `on:` field. Old `metrics/embedding=X` CLI syntax stops working; replaced by `metrics=X`.
- **`input_space` param on DSE:** Removed. Replaced by `on:` field.
- **Sampling configs:** Moved from `configs/metrics/sampling/` to `configs/sampling/`.
- **Metric signature:** Unchanged. `fn(embeddings, dataset, module, cache)` is the protocol.
- **Result key format:** Changes from `group.metric_name` to `metric_name` (no group prefix, since groups are gone). Sweep suffixes unchanged (`metric__param_value`).
- **`evaluate_metrics()` (Hydra-free path):** Unchanged. Programmatic callers pass explicit arrays and metric names. No routing needed.
- **`run_pipeline()`:** Removed. Multi-step workflows use the Python API or sequential CLI runs from precomputed embeddings.
- **Duplicate configs removed:** `dataset/dse_knn.yaml` and `embedding/dse_knn.yaml` collapse to one `dse_knn.yaml` with `on: embedding`. For DSE on dataset space, use `on: dataset` override or `on: [embedding, dataset]` sweep.

### Validation

Both paths must produce identical results for the same metric + data + sampling:

**CLI dry-run with metric at every step:**
```bash
uv run python -m manylatents.main \
    data=swissroll algorithms/latent=pca \
    metrics=trustworthiness \
    sampling=balanced
```

**API equivalent:**
```python
from manylatents.api import run
result = run(
    data="swissroll",
    algorithms={"latent": "pca"},
    metrics={"trustworthiness": {
        "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
        "_partial_": True,
        "n_neighbors": 5,
        "on": "embedding",
    }},
    sampling={"embedding": {
        "_target_": "manylatents.utils.sampling.RandomSampling",
        "fraction": 0.1,
    }},
)
```

Both paths go through `evaluate_outputs()` and must produce the same scores for the same inputs. Integration tests verify this parity.

### Config Structure (Final)

The 3 metric subdirectories are removed. All metrics live in a flat `configs/metrics/` directory. Every metric config has an explicit `on:` field. Hydra composition uses the single `metrics` config group.

```
configs/
  metrics/
    null.yaml
    noop.yaml
    trustworthiness.yaml       # on: embedding
    continuity.yaml            # on: embedding
    knn_preservation.yaml      # on: embedding
    dse_knn.yaml               # on: embedding (default)
    spectral_gap_ratio.yaml    # on: module
    geodesic_distance_correlation.yaml  # on: dataset
    ...
  sampling/                    # NEW top-level config group
    balanced.yaml
    random.yaml
    stratified.yaml
    farthest_point.yaml
```

Individual configs define one metric each. Bundle configs compose multiple metrics via Hydra defaults:

```yaml
# configs/metrics/trustworthiness.yaml (individual)
trustworthiness:
  _target_: manylatents.metrics.trustworthiness.Trustworthiness
  _partial_: True
  n_neighbors: 5
  on: embedding

# configs/metrics/standard.yaml (bundle)
defaults:
  - trustworthiness
  - continuity
  - knn_preservation
  - spectral_gap_ratio
  - _self_
```

CLI usage:

```bash
# Single metric
uv run python -m manylatents.main metrics=trustworthiness data=swissroll algorithms/latent=pca

# Bundle (multiple metrics)
uv run python -m manylatents.main metrics=standard data=swissroll algorithms/latent=pca

# Bundle + parameter override
uv run python -m manylatents.main metrics=standard metrics.trustworthiness.n_neighbors=25
```

Bundles replace the old 3-subdirectory composition model. Unlike the old structure, a single bundle can mix metrics from any `on` target (embedding, dataset, module) — no need to compose across subgroups.

`flatten_and_unroll_metrics()` simplifies from two-level iteration (group → metric) to single-level:

```python
# New: flat iteration, no group prefix
for metric_name, metric_cfg in all_metrics.items():
    if not (isinstance(metric_cfg, DictConfig) and "_target_" in metric_cfg):
        continue
    flat[metric_name] = metric_cfg  # result key = metric_name directly
```

Result keys become `trustworthiness`, `dse_knn__on_dataset`, etc. (no group prefix).

`config.yaml` changes:

```yaml
# Old (removed)
- metrics: null
- metrics/embedding: null
- metrics/module: null
- metrics/dataset: null
- pipeline: null
pipeline: []

# New
- metrics: null
- optional sampling: null
```

### What Does NOT Change

- Metric protocol signature: `fn(embeddings, dataset, module, cache)`
- `SamplingStrategy.sample()` method (kept for backward compat alongside new `get_indices()`)
- Callback interface (`on_latent_end` receives full unsampled outputs + scores)
- `evaluate_metrics()` in evaluate.py (Hydra-free path)
- Content-addressed caching via `_content_key()`

### What Gets Removed

- `run_pipeline()` from experiment.py (~200 lines)
- Pipeline routing branch in `api.py` (lines 169-174) — `api.run()` always calls `run_algorithm()`
- `pipeline: []` from `config.yaml` and `- pipeline: null` from defaults
- `SaveTrajectory` callback (pipeline-specific, `callbacks/embedding/save_trajectory.py`)
- `_DATA_KNN_METRICS` and `_SPECTRAL_METRICS` frozensets
- `configs/metrics/default.yaml`
- `configs/metrics/embedding/`, `configs/metrics/dataset/`, `configs/metrics/module/` subdirectories
- `configs/metrics/sampling/` (moved to `configs/sampling/`)
- `input_space` parameter on DSE configs
- Subgroup defaults lines in `config.yaml`

## Scope

- `run_algorithm()` in experiment.py: full redesign of metric routing, sampling, and cache prewarming
- `evaluate_outputs()`: rewritten to use `on` field routing and per-output sampling
- `flatten_and_unroll_metrics()`: rewritten for flat config structure, `on` as sweepable field
- `extract_k_requirements()` and `prewarm_cache()`: rewritten to derive from `on` values
- All ~50 metric configs: rewritten as flat files with explicit `on:` field
- Sampling configs: moved to top-level `configs/sampling/`
- `run_pipeline()`: removed (multi-step via API or sequential CLI)
- `config.yaml`: simplified (single `metrics` group, new `sampling` group)
- CLI + API parity: integration tests verifying identical results from both paths

## Known Tension

The metric protocol signature uses `embeddings` as the parameter name for primary data. Under routing, this parameter receives whatever `on` points to — which might be `dataset.data` or an `affinity` matrix, not embeddings. The name becomes misleading but the function works. Option 1 (accept the misnomer) was chosen for the initial implementation.

## Asymmetry: Metric Routing vs Sampling

Metric routing via `"on"` is fully dynamic — any key in the `outputs` dict is a valid target, and new outputs are automatically available via `module.extra_outputs()`.

Sampling is NOT dynamic. It has two fixed integration points hardcoded in the pipeline:

- `sampling.dataset` — wired in `run_algorithm()` before `execute_step()`
- `sampling.embedding` — wired in `evaluate_outputs()` before the metric loop

Other sampling keys (e.g., `sampling.affinity`) would be silently ignored. Adding a new sampling position requires adding an integration point in the pipeline code. This asymmetry exists because sampling affects pipeline execution order (pre-fit vs post-fit), while metric routing only affects which data is read at evaluation time.

This asymmetry is documented in `docs/metrics.md` under "Pipeline Execution Model."

## Commit Plan

Each commit is independently testable — tests pass at every step. Ordered by dependency.

### Commit 1: Remove `run_pipeline()`
Standalone cleanup. No dependency on other commits.
- Delete `run_pipeline()` from `experiment.py` (~270 lines)
- Remove pipeline routing from `api.py` (lines 169-174, import on line 38)
- Remove `pipeline: []` and `- pipeline: null` from `config.yaml`
- Delete `SaveTrajectory` callback (`callbacks/embedding/save_trajectory.py`)
- Remove `test_step_snapshots.py` pipeline tests
- Update `callbacks/embedding/__init__.py` (remove SaveTrajectory export)
- Update docs: `evaluation.md` (remove Pipeline Mode section)

### Commit 2: Normalize `extra_outputs()` keys
Standalone. No dependency on other commits.
- Rename `LatentModule` methods: `affinity_matrix()` → `affinity()`, `kernel_matrix()` → `kernel()`, `adjacency_matrix()` → `adjacency()`
- Old names kept as deprecated aliases (call through to new names)
- Update `extra_outputs()` in base class and all subclasses to return short keys
- Update `run_algorithm()` line 662-667 where extras are attached
- Update `resolve_matrix()` in `utils/metrics.py`

### Commit 3: Add `get_indices()` to sampler implementations
Standalone. No dependency on other commits.
- Add `get_indices()` to `SamplingStrategy` protocol
- Implement on `StratifiedSampling`, `FarthestPointSampling`, `FixedIndexSampling` (already exists on `RandomSampling`)
- Unit tests for each implementation

### Commit 4: Rewrite metric configs — flat structure with `on` field
The big config change. Tests break until evaluate_outputs catches up (commit 5).
- Move all ~50 metric configs from `embedding/`, `dataset/`, `module/` to flat `configs/metrics/`
- Add explicit `on:` field to every config
- Remove duplicate configs (e.g., `dataset/dse_knn.yaml` — collapsed into `dse_knn.yaml` with `on: embedding`)
- Create bundle configs: `standard.yaml`, `noop.yaml` (updated), `null.yaml` (updated)
- Delete `configs/metrics/embedding/`, `configs/metrics/dataset/`, `configs/metrics/module/` directories and their `__init__.py` files
- Delete `configs/metrics/default.yaml`
- Delete `configs/metrics/sampling/` (moved in commit 6)
- Update `config.yaml`: remove subgroup defaults, keep single `- metrics: null`
- Remove `input_space` parameter from `diffusion_spectral_entropy.py`
- Update experiment configs: `eval_algorithm.yaml`, `backend_comparison.yaml`, `torchdr_parameter_sweep.yaml`, `torchdr_profiling.yaml`

### Commit 5: Rewrite `evaluate_outputs()` and cache prewarming
Core routing change. Depends on commit 4.
- Rewrite `flatten_and_unroll_metrics()` for single-level iteration, `on` as sweepable field
- Rewrite `evaluate_outputs()`: build `outputs` dict, route metrics by `on` value, pop `on` before instantiation, loud skip on missing output
- Delete `_DATA_KNN_METRICS` and `_SPECTRAL_METRICS` frozensets
- Rewrite `extract_k_requirements()`: group by `on` value, derive cache needs from config
- Rewrite `prewarm_cache()`: iterate per-output kNN requirements, eigenvalue prewarming for `on: module`

### Commit 6: Sampling config migration and pre-fit sampling
Depends on commits 3 and 5.
- Create `configs/sampling/` top-level config group with `random.yaml`, `stratified.yaml`, `farthest_point.yaml`
- Add `- optional sampling: null` to `config.yaml`
- Implement pre-fit sampling in `run_algorithm()` (between tensor extraction and execute_step)
- Update `evaluate_outputs()` to read sampling from `cfg.sampling` instead of `cfg.metrics.sampling`
- Index propagation: embedding indices applied to dataset for cross-space metrics

### Commit 7: Update docs, CI, tests
Depends on all previous commits.
- Update CLAUDE.md: CLI examples, config directory listings
- Update README.md: CLI examples, config patterns
- Update CONTRIBUTING.md: new metric guide with `on:` field
- Update `docs/metrics.md`, `docs/testing.md`, `docs/extensions.md`
- Rewrite `docs/macros.py` `_metrics_table()` for flat structure
- Rewrite `scripts/check_docs_coverage.py` for flat structure
- Rewrite `.github/workflows/scripts/test_metrics.sh` for flat structure
- Update `utils/merge_results.py` default column (no group prefix)
- Update `tests/test_docs_macros.py`
- Update `tests/callbacks/test_save_trajectory.py` (if kept post pipeline removal)

### Commit 8: Integration tests — CLI + API parity
Final validation. Depends on all previous commits.
- Test: CLI `metrics=trustworthiness` produces correct scores
- Test: API `run(metrics={"trustworthiness": {..., "on": "embedding"}})` produces same scores
- Test: `on: [embedding, dataset]` sweep produces two evaluations
- Test: `on: module` correctly routes to module-level metrics
- Test: `on: affinity` against PCA skips with warning
- Test: `sampling.dataset` subsamples before fit
- Test: `sampling.embedding` subsamples before eval
- Test: Cross-space metrics get matching indices from embedding sampler
- Test: Cache isolation — different `on` values don't share cache entries
- Test: Bundle configs compose correctly

## Full Change Manifest

| File | Commit | Action |
|---|---|---|
| `experiment.py` (run_pipeline) | 1 | Delete ~270 lines |
| `api.py` (pipeline routing) | 1 | Remove import + branch |
| `configs/config.yaml` (pipeline) | 1 | Remove pipeline defaults |
| `callbacks/embedding/save_trajectory.py` | 1 | Delete file |
| `tests/test_step_snapshots.py` | 1 | Delete pipeline tests |
| `docs/evaluation.md` | 1 | Remove pipeline section |
| `algorithms/latent/latent_module_base.py` | 2 | Rename matrix methods |
| `algorithms/latent/*.py` (subclasses) | 2 | Update extra_outputs keys |
| `utils/metrics.py` (resolve_matrix) | 2 | Update method names |
| `experiment.py` (extra_outputs attachment) | 2 | Update key names |
| `utils/sampling.py` | 3 | Add get_indices to all strategies |
| `configs/metrics/**/*.yaml` (~50 files) | 4 | Move to flat, add `on:` |
| `configs/metrics/__init__.py` files | 4 | Delete subdir inits |
| `configs/metrics/default.yaml` | 4 | Delete |
| `configs/experiment/*.yaml` (4 files) | 4 | Update override syntax |
| `metrics/diffusion_spectral_entropy.py` | 4 | Remove `input_space` |
| `utils/metrics.py` (flatten_and_unroll) | 5 | Rewrite for flat + `on` |
| `experiment.py` (evaluate_outputs) | 5 | Rewrite routing + cache |
| `experiment.py` (frozensets) | 5 | Delete |
| `configs/sampling/*.yaml` | 6 | New config group |
| `configs/config.yaml` (sampling) | 6 | Add sampling defaults |
| `experiment.py` (run_algorithm pre-fit) | 6 | Add pre-fit sampling |
| `CLAUDE.md` | 7 | Update examples |
| `README.md` | 7 | Update examples |
| `CONTRIBUTING.md` | 7 | Update metric guide |
| `docs/metrics.md` | 7 | Rewrite |
| `docs/testing.md` | 7 | Update examples |
| `docs/extensions.md` | 7 | Update examples |
| `docs/macros.py` | 7 | Rewrite metrics_table |
| `scripts/check_docs_coverage.py` | 7 | Rewrite for flat |
| `.github/workflows/scripts/test_metrics.sh` | 7 | Rewrite |
| `utils/merge_results.py` | 7 | Update default column |
| `tests/test_docs_macros.py` | 7 | Update assertion |
| Integration tests (new) | 8 | Full parity suite |

## Open Questions

None. All decisions locked in via brainstorming session.
