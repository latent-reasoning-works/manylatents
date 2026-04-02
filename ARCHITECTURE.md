# Architecture Overview

Manylatents is a unified dimensionality reduction and neural network analysis framework.
It wraps 20+ algorithms behind two base classes, evaluates them with 40+ metrics through
a shared cache, and orchestrates everything via Hydra configs or a Hydra-free Python API.

This document is a coarse map of the codebase — read it once, then use symbol search to
find specifics. Revisit a couple times a year; not every commit.

## High-Level Data Flow

```
                          ┌──────────────┐
User ───► CLI (main.py)   │  Hydra layer │   instantiates configs
     or   API (api.py) ───┤  (optional)  ├─► resolves algorithms, data, metrics
                          └──────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ experiment.py  │   the engine
                        │ run_experiment │
                        └───────┬────────┘
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              LatentModule  Lightning   Precomputed
              fit/transform  Trainer     loader
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                        ┌────────────────┐
                        │  evaluate.py   │   metric evaluation
                        │  evaluate()    │   with shared kNN/SVD cache
                        └───────┬────────┘
                                ▼
                          LatentOutputs
                          {"embeddings": ..., "scores": ...}
```

## Project Structure

```
manylatents/
├── main.py                  # Hydra CLI entry point (@hydra.main)
├── api.py                   # Hydra-free Python API (run())
├── experiment.py            # Engine: run_experiment() — orchestrates fit → evaluate
├── evaluate.py              # Unified evaluate(), cache pre-warming, metric dispatch
│
├── algorithms/
│   ├── latent/              # LatentModule implementations (fit/transform)
│   │   ├── latent_module_base.py   # ABC: LatentModule
│   │   ├── pca.py, umap.py, tsne.py, phate.py, diffusion_map.py, ...
│   │   ├── foundation_encoder.py   # Wraps pretrained models as LatentModules
│   │   ├── reeb_graph.py           # Topological methods
│   │   ├── leiden.py, classifier.py, selective_correction.py
│   │   └── merging.py              # Module composition
│   └── lightning/           # LightningModule implementations (trainer.fit)
│       ├── reconstruction.py       # Autoencoder-based DR
│       ├── latent_ode.py           # Neural ODE latent dynamics
│       ├── losses/                 # MSELoss, geometric losses
│       └── networks/               # Autoencoder, AANet, LatentODE architectures
│
├── metrics/
│   ├── metric.py            # Metric protocol (function signature contract)
│   ├── registry.py          # @register_metric, MetricSpec, aliases, compute_metric()
│   ├── trustworthiness.py, continuity.py, knn_preservation.py, ...
│   ├── persistent_homology.py, fractal_dimension.py          # topological
│   ├── diffusion_spectral_entropy.py, spectral_gap_ratio.py  # spectral
│   └── ... (40+ metric modules)
│
├── data/
│   ├── capabilities.py      # Protocol-based dataset capability detection
│   ├── swissroll.py, torus.py, gaussian_blobs.py, ...  # synthetic datasets
│   ├── precomputed_datamodule.py   # Load pre-saved embeddings
│   ├── reasoning_trace.py, text.py # non-geometric data sources
│   └── synthetic_dataset.py        # Base for synthetic data generation
│
├── callbacks/
│   ├── embedding/           # Post-embedding hooks
│   │   ├── base.py          # EmbeddingCallback ABC, LatentOutputs type
│   │   ├── plot_embeddings.py, save_outputs.py, wandb_log_scores.py
│   │   └── loadings_analysis.py
│   └── diffusion_operator.py  # Diffusion-specific callbacks
│
├── configs/                 # Hydra config groups
│   ├── __init__.py          # SearchPathPlugin registration
│   ├── algorithms/latent/   # One YAML per LatentModule
│   ├── algorithms/lightning/ # One YAML per LightningModule
│   ├── data/                # Dataset configs
│   ├── metrics/             # Metric configs (_partial_: True, at: field)
│   ├── callbacks/           # Callback configs
│   ├── cluster/             # SLURM cluster profiles (mila, narval, etc.)
│   └── resources/           # GPU/CPU resource presets
│
├── utils/
│   ├── metrics.py           # compute_knn() (FAISS-GPU cache), compute_svd_cache()
│   ├── backend.py           # TorchDR interop, torchdr_knn_to_dense()
│   ├── sampling.py          # Pre-fit data sampling strategies
│   ├── data.py              # Data loading and format detection
│   ├── knn.py               # kNN utilities beyond FAISS
│   ├── kernel_utils.py      # Kernel construction helpers
│   └── utils.py             # Logging, path helpers, misc
│
├── lightning/               # Lightning-specific infrastructure
│   └── callbacks/           # Trainer-level callbacks (checkpointing, etc.)
│
tests/                       # ~60 test files, ~489 test functions
scripts/                     # check_docs_coverage.py, helpers
docs/                        # mkdocs source (deployed to GitHub Pages)
```

## Core Components

### Two Algorithm Bases

Every algorithm that takes data and produces output goes through one of two base classes.
The decision is about API surface, not whether gradients are involved.

**LatentModule** — self-contained algorithms with `fit(x)` / `transform(x)`. The base
class lives in `algorithms/latent/latent_module_base.py`. Base constructor accepts
`n_components`, `init_seed`, `backend`, `device`, `neighborhood_size`. Subclasses
accept `random_state` (sklearn convention) and map it to `init_seed` when calling
`super().__init__()`. Subclasses store `_is_fitted` state. Input/output can be numpy
or torch — helpers `_to_numpy()` and `_to_output()` handle conversion.

**LightningModule** — algorithms that need the Lightning Trainer (callbacks, logging,
checkpointing, multi-GPU). Live in `algorithms/lightning/`. Interface: `setup()`,
`training_step()`, `encode(x)`, `configure_optimizers()`.

### Metric System

Metrics follow a function protocol: `(embeddings, dataset?, module?, cache?, <named_params>) → float | tuple[float, np.ndarray] | dict`. Each metric declares its own named parameters (e.g. `k`, `n_neighbors`) — no `**kwargs` catch-all.

The registry (`metrics/registry.py`) provides `@register_metric` for auto-discovery,
`MetricSpec` for aliases with preset params (e.g., "beta_0" → persistent_homology with
`homology_dim=0`), and `compute_metric()` / `get_metric()` for programmatic access.

All metrics share a `cache` dict passed through the call chain. The cache holds
deduplicated kNN graphs and eigenvalue decompositions, pre-warmed by
`evaluate.py:prewarm_cache()` which scans metric signatures for `k` requirements.

### Data Contract

`LatentOutputs = dict[str, Any]` (defined in `callbacks/embedding/base.py`).
Required key: `"embeddings"` (n, d). Optional: `"scores"`, `"label"`, `"metadata"`.
`EmbeddingOutputs` is a deprecated alias that still exists.

### Config System

Hydra config groups under `manylatents/configs/`. `ManylatentsSearchPathPlugin` in
`configs/__init__.py` registers the package path on import — no hydra_plugins namespace
needed. Extension packages (manylatents-omics) register their own SearchPathPlugins
via `importlib.metadata` entry points, discovered in `main.py:_discover_extensions()`.

Metric configs use `_partial_: True` so Hydra produces a callable partial, and an `at:`
field for evaluation context routing. Parameter sweeps use list values in YAML —
`flatten_and_unroll_metrics()` in `utils/metrics.py` does Cartesian expansion.

## Architectural Invariants

- **Core never imports from extensions.** The `manylatents-omics` package extends via
  `pkgutil.extend_path()` — core has zero references to `manylatents.dogma`,
  `manylatents.popgen`, or `manylatents.singlecell`.

- **Shared cache, not per-metric computation.** kNN graphs and SVD decompositions are
  computed once and shared via the `cache` dict. If you add a metric that needs kNN,
  use `compute_knn()` from `utils/metrics.py`, never sklearn directly.

- **Lazy imports for optional deps.** A LatentModule file must import cleanly without its
  optional dependency installed. Heavy imports go inside methods.

- **No new top-level packages for algorithms.** Everything producing output from data
  goes through LatentModule or LightningModule.

- **Configs are the API contract.** CI smoke-tests every config YAML. If you add a
  LatentModule, you add a config — it's not optional.

## External Dependencies

| Category | Key packages |
|----------|-------------|
| Core ML | torch >=2.3, lightning >=2.5, scipy >=1.8 <1.15 |
| DR | phate, umap-learn, opentsne, multiscale-phate, archetypes |
| Config | hydra-core >=1.3, hydra-zen >=0.13 |
| Topology | ripser, gudhi |
| GPU accel | FAISS (optional), TorchDR >=0.3 (optional extra) |
| Dynamics | torchdiffeq, torchsde |

scipy upper bound (`<1.15`) is load-bearing — archetypes and PHATE break above it.

## Deployment & CI

**CI** (`.github/workflows/build.yml`): matrix over Python 3.11/3.12, runs unit tests,
CLI smoke tests per algorithm family, docs coverage check, mkdocs strict build. Docs
deploy to GitHub Pages on main push.

**Cluster**: Hydra launcher configs for SLURM clusters (Mila, Narval). `cluster=mila`
plus `resources=gpu` routes to GPU partition. All compute runs via SLURM, never login nodes.

**Package**: Published to PyPI via GitHub release workflow. `uv` for dependency management;
wheelnext uv for CUDA wheel variants (mamba-ssm, flash-attn, transformer-engine).

## Development & Testing

**Local setup**: `uv sync` (or `uv sync --extra torchdr` for GPU-accelerated DR).

**Test suites**:
- `uv run pytest tests/ -x -q` — core tests
- `uv run pytest manylatents/callbacks/tests/ -x -q` — callback tests
- `uv run pytest manylatents/lightning/callbacks/tests/` — lightning callback tests
- `uv run mkdocs build --strict` — docs build

**Code quality**: pytest, mkdocs strict mode, docs coverage script.

## Project Identification

| | |
|---|---|
| **Project** | manylatents |
| **Version** | 0.1.3 |
| **Repository** | github.com/latent-reasoning-works/manylatents |
| **Docs** | latent-reasoning-works.github.io/manylatents |
| **License** | MIT |
| **Last updated** | 2026-04-01 |
