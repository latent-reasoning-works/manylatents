# CLAUDE.md

Unified dimensionality reduction and neural network analysis. PyTorch Lightning + Hydra + uv.

## Entry Points

```bash
# CLI — primary interface
uv run python -m manylatents.main algorithms/latent=pca data=swissroll metrics/embedding=trustworthiness

# LightningModule path
uv run python -m manylatents.main algorithms/lightning=ae_reconstruction data=swissroll trainer.fast_dev_run=true

# Multirun sweep
uv run python -m manylatents.main --multirun algorithms/latent=umap,phate,tsne data=swissroll metrics/embedding=trustworthiness

# SLURM submission
uv run python -m manylatents.main -m cluster=mila resources=gpu algorithms/latent=umap data=swissroll
```

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
algorithms/latent/      pca, umap, tsne, phate, diffusionmap, mds, aa, multiscale_phate, noop, classifier
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

## Adding New Components

**New metric**: wrapper function → `@register_metric` decorator → config YAML → import in `__init__.py` → CI smoke test.
See `CONTRIBUTING.md` for the full 4-step pipeline.

**New LatentModule**: subclass `LatentModule` → implement `fit(x)` + `transform(x)` → config YAML → import in `__init__.py`.

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
