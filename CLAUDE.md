# CLAUDE.md

Unified dimensionality reduction and neural network analysis. PyTorch Lightning + Hydra + uv.

## Execution — ALWAYS through SLURM

**NEVER run compute on a login node.** All runs — including single dry-runs, validation, and sweeps — go through the submitit plugin via `cluster=mila`.

```bash
# Single run (CPU)
uv run python -m manylatents.main cluster=mila resources=cpu_light \
    algorithms/latent=pca data=swissroll metrics/embedding=trustworthiness

# Single run (GPU)
uv run python -m manylatents.main cluster=mila resources=gpu \
    algorithms/latent=umap data=swissroll

# Multirun sweep (each combo = separate SLURM job)
uv run python -m manylatents.main -m cluster=mila resources=cpu_light \
    algorithms/latent=umap,phate,tsne data=swissroll metrics/embedding=trustworthiness

# Experiment config (sweep defined in YAML)
uv run python -m manylatents.main -m experiment=artifact_grid_sweep \
    cluster=mila resources=cpu_light
```

See `shop/CLAUDE.md` for cluster configs, resource profiles, and QOS limits.

```python
from manylatents.api import run
result = run(data="swissroll", algorithms={"latent": "pca"}, metrics={"embedding": {"trustworthiness": {"_target_": "manylatents.metrics.trustworthiness.Trustworthiness", "_partial_": True, "n_neighbors": 5}}})
result["embeddings"]  # (n, d) ndarray
result["scores"]      # {"embedding.trustworthiness": 0.95}
```

## Core Abstractions

**Two algorithm base classes, one decision rule:**

| If the algorithm... | Use | Interface |
|---|---|---|
| has no training loop | `LatentModule` (`algorithms.latent.latent_module_base`) | `fit(x)` / `transform(x)` |
| trains with backprop | `LightningModule` (`algorithms.lightning.*`) | `trainer.fit()` / `encode(x)` |

**Metric protocol** (`manylatents.metrics.metric`):

```python
def metric_fn(embeddings: np.ndarray, dataset=None, module=None, cache=None) -> float | dict
```

All metrics share a `cache` dict for deduplicated kNN/eigenvalue computation. Three evaluation contexts: `embedding`, `dataset`, `module`.

**Data contract**: `EmbeddingOutputs = dict[str, Any]` — required key `"embeddings"`, optional `"scores"`, `"label"`, `"metadata"`.

## Config System

Hydra config groups under `manylatents/configs/`:

```
algorithms/latent/      pca, umap, tsne, phate, diffusionmap, mds, aa, multiscale_phate, noop, classifier, leiden
algorithms/lightning/   ae_reconstruction, aanet_reconstruction, latent_ode, hf_trainer
data/                   swissroll, torus, saddle_surface, gaussian_blobs, dla_tree, precomputed, test_data
metrics/embedding/      trustworthiness, continuity, knn_preservation, persistent_homology, fractal_dimension, anisotropy, ...
metrics/dataset/        stratification, admixture_laplacian, geodesic_distance_correlation
metrics/module/         spectral_gap_ratio, spectral_decay_rate, affinity_spectrum, connected_components, ...
callbacks/embedding/    default, minimal, save_embeddings, wandb_log_scores
logger/                 none, wandb
cluster/                mila, mila_remote, narval
```

Metric configs use `_partial_: True` with nested structure:
```yaml
trustworthiness:
  _target_: manylatents.metrics.trustworthiness.Trustworthiness
  _partial_: True
  n_neighbors: 25
```

## Experiment Configs

Experiment configs live in `manylatents/configs/experiment/` and bundle data + algorithm + metrics + sweep params into a single YAML. Submit via:

```bash
uv run python -m manylatents.main -m experiment=<name> cluster=mila resources=cpu_light
```

| Config | What it sweeps | Jobs |
|--------|---------------|------|
| `artifact_grid_sweep` | UMAP: neighborhood_size(6) × min_dist(5) × seed(3) | 90 |
| `artifact_leiden_sweep` | Leiden: neighborhood_size(6) × seed(3) | 18 |
| `torchdr_parameter_sweep` | TorchDR backends × algorithms | varies |

Artifact sweeps require preprocessed data — run `scripts/prep_embryoid_body_pca.py` first.

## Adding New Components

**New metric**: wrapper function → `@register_metric` decorator → config YAML → import in `__init__.py` → CI smoke test.
See `CONTRIBUTING.md` for the full 4-step pipeline.

**New LatentModule** — there are exactly 4 files to touch:

1. **Module**: `manylatents/algorithms/latent/<name>.py`
   - Subclass `LatentModule`
   - Accept `n_components`, `random_state`, `neighborhood_size`, `backend`, `device`, `**kwargs` and pass them to `super().__init__()`
   - `neighborhood_size` overrides any module-specific neighbor count (e.g. `self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors`)
   - `fit(x: Tensor)` — fit on data, set `self._is_fitted = True`
   - `transform(x: Tensor) -> Tensor` — return embeddings
   - Use shared infra: `compute_knn()` from `utils/metrics.py` (FAISS-GPU cache), not sklearn directly
   - Keep third-party imports **lazy** (inside methods) if the dep is optional — the module file must import cleanly without the optional dep installed

2. **Export**: `manylatents/algorithms/latent/__init__.py`
   - Add `from .<name> import <Name>Module` and add to `__all__`
   - No try/except guard — lazy imports in the module file handle missing optional deps

3. **Hydra config**: `manylatents/configs/algorithms/latent/<name>.yaml`
   - `_target_: manylatents.algorithms.latent.<name>.<Name>Module`
   - Include `random_state: ${seed}` and `neighborhood_size: ${neighborhood_size}`
   - Set `backend: null` and `device: null` for TorchDR-capable modules

4. **Test**: `tests/test_<name>.py`
   - Use `pytest.importorskip("<optional_dep>")` at module level for optional deps
   - Test `fit_transform()` returns correct shape
   - Test the module `isinstance(m, LatentModule)`
   - Test determinism (same seed → same output)

**Do NOT** create new top-level packages (like `analysis/`) for algorithms. Everything that takes data and produces an output goes through the LatentModule or LightningModule interface.

**New LightningModule**: subclass → implement `setup()` + `training_step()` + `encode()` + `configure_optimizers()` → config YAML.

## Key Files

| File | What it does |
|------|-------------|
| `main.py` | CLI entry point |
| `api.py` | Python API (`run()`) |
| `experiment.py` | Core engine: `run_algorithm()`, `evaluate_embeddings()`, `prewarm_cache()` |
| `metrics/metric.py` | Metric protocol definition |
| `metrics/registry.py` | `@register_metric` decorator, `list_metrics()` |
| `utils/metrics.py` | `compute_knn()`, `compute_svd_cache()` — shared cache infrastructure |
| `configs/__init__.py` | Hydra SearchPathPlugin + ConfigStore registration |
| `data/capabilities.py` | Dataset capability detection |

## Gotchas

- **`uv run`, not `python`** — always prefix with `uv run` or activate the venv.
- **`scipy>=1.8,<1.15`** — pinned for archetypes/PHATE. Don't relax without testing.
- **Hydra null** — CLI doesn't support `callbacks=null`. Use `logger=none` or explicit null configs.
- **API metrics need `_target_`** — empty dicts `{}` are silently skipped by `flatten_and_unroll_metrics()`.
- **LightningModule unit tests** — must call `model.setup()` if not using `trainer.fit()`.
- **`save_hyperparameters`** — always `ignore=["datamodule", "network", "loss"]`.
- **Loss functions** — use project's `MSELoss` (`outputs, targets, **kwargs`), not `torch.nn.MSELoss`.

## Tests

```bash
uv run pytest tests/ -x -q                          # core tests (171 pass)
uv run pytest manylatents/callbacks/tests/ -x -q     # callback tests
uv run pytest manylatents/lightning/callbacks/tests/  # lightning callback tests
uv run mkdocs build --strict                         # docs build
uv run python3 scripts/check_docs_coverage.py        # config coverage
```

## Namespace Extensions

`manylatents-omics` extends via `pkgutil.extend_path()`. Adds `manylatents.dogma` (foundation encoders), `manylatents.popgen` (population genetics), `manylatents.singlecell` (AnnData). Core never imports from extensions.
