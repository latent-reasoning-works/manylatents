# CLAUDE.md

Agent instructions for manyLatents. Read this first, then MEMORY.md.

## Project Identity

manyLatents is a unified framework for dimensionality reduction and neural network analysis. Built on PyTorch Lightning + Hydra.

**Two execution modes:**
- **CLI** (`python -m manylatents.main`) — single-step execution: one algorithm + metrics on one dataset. This is the primary user-facing interface.
- **Python API** (`manylatents.api.run()`) — programmatic interface that enables multi-step workflows when called directly by other agents or scripts. Supports `input_data` for chaining results between calls and `pipeline` configs for sequential steps.

**manyLatents is NOT:**
- An orchestration framework (future: manyAgents handles multi-step orchestration by calling the API)
- An RL training system (future: Geomancer)
- A cluster job manager (that's Shop)

## Architecture at a Glance

### Two Algorithm Base Classes

- **`LatentModule`** (`manylatents.algorithms.latent.latent_module_base`) — fit/transform for non-neural algorithms (PCA, UMAP, t-SNE, PHATE, DiffusionMap, MDS, Archetypes, MultiscalePHATE). Subclass this for any algorithm that doesn't need a training loop. The **FoundationEncoder pattern** is a LatentModule where `fit()` is a no-op and `transform()` wraps a pretrained model. It is NOT a separate base class — it is a usage convention within LatentModule for frozen/pretrained encoders.
- **`LightningModule` subclasses** (`manylatents.algorithms.lightning`) — neural networks with Lightning training loops. Subclass this for autoencoders, VAEs, Latent ODEs, or any trainable architecture.

**Decision rule:** If the algorithm trains with backprop, use LightningModule. If not, use LatentModule. Frozen foundation models are LatentModules.

### Data Contract

`EmbeddingOutputs = dict[str, Any]` — the universal interchange format. This is a HANDOFF INTERFACE.

Required key: `"embeddings"`. Optional keys: `"label"`, `"metadata"`, `"scores"`.

It is a dict (not a dataclass) because stateless agents and downstream consumers must read/write it without schema migration. When a new metric injects a custom field, every downstream consumer still works.

### Metric Contract

Metrics follow the `Metric` protocol (`manylatents.metrics.metric`):

```python
def __call__(
    self,
    embeddings: np.ndarray,
    dataset=None,
    module=None,
    _knn_cache=None,
) -> float | tuple[float, np.ndarray] | dict[str, Any]
```

Three evaluation contexts: `dataset`, `embedding`, `module`. Registered via Hydra `_target_` with `_partial_: True`. Parameters (like `return_per_sample`, `k`, `n_neighbors`) are set in Hydra config, not at call time.

`flatten_and_unroll_metrics()` handles nested/swept metric configs. List-valued parameters expand via Cartesian product.

### Namespace Extension Pattern

`manylatents-omics` extends via `pkgutil.extend_path()` in `__init__.py`. Extensions add algorithms, metrics, and data modules to the same namespace. Core never imports from extensions; extensions import from core.

## Handoff Interfaces

### Starting a Session

1. Read this file (CLAUDE.md) — understand the rules
2. Read MEMORY.md — understand current state, known results, known gotchas
3. Check for any consolidated results or EmbeddingOutputs from previous runs

### Ending a Session

1. Update MEMORY.md with: what was done, what was learned, what's pending
2. Ensure any new scripts have explicit input/output declarations in docstrings
3. Do NOT leave implicit state — if the next session needs to know something, write it down

### Compute Handoff (SLURM)

- You cannot SSH into compute nodes or wait for jobs
- Workflow: write scripts -> user runs `sbatch` (or Shop launcher) -> you pick up results next session
- Every submission script must be self-documenting: partition, GPU count, venv path, exact command
- For Shop launcher configs, reference `shop/CLAUDE.md`

## Safety Constraints (Non-Negotiable)

These are HARD RULES. Violating them can get users locked out of compute clusters.

- **NEVER** run computation on login nodes. Every compute command goes through `sbatch`, `srun`, or `salloc`. If you generate a script that runs `python train.py` directly on a login node, that is a FATAL mistake.
- **ALWAYS** set time limits. Default to 15 minutes for test jobs. Never exceed 4 hours without explicit user confirmation.
- **ALWAYS** use `fast_dev_run=true` or `max_epochs<=2` for testing. Unbounded training loops are forbidden in test contexts.
- **ALL** outputs go to `$SCRATCH` (on Mila) or a designated output directory. Never write to `$HOME` on cluster systems.
- **ALWAYS** set `WANDB_MODE=offline` for test runs. Use `logger=none` for CI. Only go online when the run config is verified.
- **NEVER** run `pip install` outside a venv. Use `uv`, not `pip`, for dependency management.
- If you encounter a CAPTCHA, authentication prompt, or cluster access issue, **STOP** and inform the user.

## Key File Locations

### Core Package

| Location | Purpose | Config Group |
|----------|---------|-------------|
| `manylatents/algorithms/latent/` | LatentModule algorithms | `algorithms/latent` |
| `manylatents/algorithms/lightning/` | LightningModule algorithms | `algorithms/lightning` |
| `manylatents/metrics/` | Evaluation metrics | `metrics/embedding`, `metrics/dataset`, `metrics/module` |
| `manylatents/data/` | Data modules & datasets | `data` |
| `manylatents/callbacks/` | Embedding & trainer callbacks | `callbacks/embedding` |
| `manylatents/experiment.py` | Core engine: `run_algorithm()`, `execute_step()`, `evaluate()` |
| `manylatents/api.py` | Programmatic API: `run()` for agent-driven multi-step workflows |
| `manylatents/main.py` | CLI entry point: `python -m manylatents.main` |
| `manylatents/configs/` | Hydra config root |
| `manylatents/plugins/search_path.py` | Hydra `SearchPathPlugin` registration |

### Hydra Config Groups

```
configs/
  algorithms/
    latent/         pca, umap, tsne, phate, diffusionmap, mds, aa, noop, classifier, multiscale_phate
    lightning/      ae_reconstruction, aanet_reconstruction, latent_ode, hf_trainer
      loss/         default, ae_dim, ae_neighbors, ae_shape
      network/      autoencoder, aanet
      optimizer/    adam
  data/             swissroll, torus, saddle_surface, gaussian_blobs, clusters, dla_tree, precomputed, test_data, ...
  metrics/
    embedding/      trustworthiness, continuity, participation_ratio, fractal_dimension, knn_preservation, local_intrinsic_dimensionality, anisotropy, magnitude_dimension, persistent_homology, tangent_space, pearson_correlation, diffusion_curvature
    dataset/        admixture_laplacian, stratification, sample_id, test_metric
    module/         affinity_spectrum, connected_components, kernel_matrix_sparsity, kernel_matrix_density, diffusion_map_correlation
    sampling/       random, stratified, farthest_point
  callbacks/embedding/  default, minimal, save_embeddings, plot_embeddings, wandb_log_scores
  experiment/       single_algorithm, single_algorithm_no_metrics, eval_algorithm
  trainer/          default
  logger/           none, wandb
  cluster/          mila, mila_remote, narval
  launcher/         basic, cpu_job, gpu_job, mila_cluster, mila_cpu_cluster, cc_cpu, cc_gpu
```

### Namespace Extensions (manylatents-omics)

| Location | Purpose |
|----------|---------|
| `manylatents/dogma/encoders/` | Foundation model encoders (Evo2, ESM3, Orthrus, AlphaGenome) |
| `manylatents/dogma/algorithms/` | Fusion algorithms (CentralDogmaFusion, BatchEncoder) |
| `manylatents/dogma/data/` | Sequence & ClinVar data modules |
| `manylatents/popgen/data/` | ManifoldGeneticsDataModule |
| `manylatents/popgen/metrics/` | GeographicPreservation, AdmixturePreservation |
| `manylatents/singlecell/data/` | AnnDataModule for .h5ad files |

### Shop Integration

| Location | Purpose |
|----------|---------|
| `shop/shop/hydra/config_templates/cluster/` | Cluster configs (mila, narval, cedar) |
| `shop/shop/hydra/config_templates/resources/` | Resource templates (cpu, gpu) |
| `shop.hydra.launchers.RemoteSlurmLauncher` | Remote SLURM submission via SSH |

## Common Tasks

### Add a new metric

1. Create `manylatents/metrics/your_metric.py` with a function matching the `Metric` protocol
2. Create `manylatents/configs/metrics/embedding/your_metric.yaml` (or `dataset/` or `module/`)
3. Add `_target_`, `_partial_: True`, and default parameters
4. Import in `manylatents/metrics/__init__.py`
5. Verify: `python -c "from manylatents.metrics import YourMetric"`

### Add a new LatentModule algorithm

1. Create `manylatents/algorithms/latent/your_algo.py` inheriting from `LatentModule`
2. Implement `fit(x, y=None)` and `transform(x)`
3. Create `manylatents/configs/algorithms/latent/your_algo.yaml` with `_target_`
4. Import in `manylatents/algorithms/latent/__init__.py`
5. Test: `python -m manylatents.main algorithms/latent=your_algo data=swissroll`

### Add a new LightningModule algorithm

1. Create `manylatents/algorithms/lightning/your_algo.py` inheriting from `LightningModule`
2. Implement `setup()`, `training_step()`, `encode()`, and `configure_optimizers()`
3. Use `self.save_hyperparameters(ignore=["datamodule", "network", "loss"])`
4. Create config in `manylatents/configs/algorithms/lightning/your_algo.yaml`
5. Test: `python -m manylatents.main algorithms/lightning=your_algo data=swissroll trainer.fast_dev_run=true`

### Run an experiment

```bash
# LatentModule
python -m manylatents.main algorithms/latent=pca data=swissroll metrics/embedding=trustworthiness

# LightningModule
python -m manylatents.main algorithms/lightning=ae_reconstruction data=swissroll trainer.max_epochs=10

# With SLURM (via Shop)
python -m manylatents.main -m cluster=mila resources=gpu algorithms/latent=umap data=swissroll
```

### Use the Python API (for agent-driven multi-step workflows)

```python
from manylatents.api import run

# Single step
result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 10}},
    metrics={'embedding': {'trustworthiness': {
        '_target_': 'manylatents.metrics.trustworthiness.Trustworthiness',
        '_partial_': True, 'n_neighbors': 5
    }}}
)
embeddings = result['embeddings']
scores = result['scores']

# Chaining: feed one step's output into the next
result2 = run(
    input_data=result['embeddings'],
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.phate.PHATEModule', 'n_components': 2}}
)
```

### Test namespace extension

```bash
python -c "import manylatents; import manylatents.popgen; print('Namespace OK')"
```

## Known Gotchas

**GOTCHA: FoundationEncoder is a LatentModule pattern, NOT a separate class** -> FoundationEncoder is a usage convention within LatentModule where `fit()` is a no-op and `transform()` wraps a pretrained model. Implementations live in `manylatents-omics/manylatents/dogma/encoders/`. There is no separate FoundationEncoder base class in core.

**GOTCHA: Hydra does NOT support `null` as a CLI override value** -> You cannot do `callbacks=null` on the command line. Use explicit null config files (e.g., `metrics=null`) or `logger=none` to disable logging.

**GOTCHA: API metrics require full `_target_` configs** -> When calling `manylatents.api.run()`, metric configs MUST include `_target_` keys. Empty dicts `{}` will be silently skipped. The `flatten_and_unroll_metrics()` function filters for configs with `_target_`.

**GOTCHA: GlobalHydra conflicts** -> If running multiple Hydra calls in the same process (e.g., via the API), you must clear GlobalHydra between calls. The API handles this internally, but be aware if writing custom scripts.

**GOTCHA: `save_hyperparameters` warnings** -> Always ignore nn.Module args: `self.save_hyperparameters(ignore=["datamodule", "network", "loss"])`. Lightning can't serialize these.

**GOTCHA: LightningModule tests need `model.setup()`** -> Unit tests that don't use `trainer.fit()` must call `model.setup()` manually to initialize the network.

**GOTCHA: `scipy` upper bound** -> `scipy>=1.8,<1.15` is required for archetypes/PHATE compatibility. Don't relax this without testing.

**GOTCHA: Config group rename pending** -> `algorithms/latent/` should be renamed to `algorithms/latent_module/` per TODO in `configs/algorithms/latent/__init__.py`. Not yet done.

**GOTCHA: Loss functions** -> Use the project's `MSELoss` from `manylatents.algorithms.lightning.losses.mse` (accepts `outputs, targets, **kwargs`), NOT `torch.nn.MSELoss`.

## UV Dependency Management

Trust the resolver. Use loose lower bounds. Don't pin torch versions. Don't duplicate transitive dependencies. Run `uv sync` and let it figure out compatibility. See the project root `CLAUDE.md` for detailed UV policy.
