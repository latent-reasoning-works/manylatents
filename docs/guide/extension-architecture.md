# Extension Architecture

How manyLatents is structured for extensibility, and why.

## Design Philosophy

manyLatents is built around a simple idea: **every interface between stages is a file with a known schema**. This matters because the agents and scripts that compose manyLatents into larger workflows are stateless — they don't remember what happened in the last call. If the output of one step doesn't fully describe itself, the next step can't use it.

This constraint shapes everything:

- **EmbeddingOutputs** is a `dict[str, Any]`, not a dataclass. When a new metric injects a custom field, every downstream consumer still works without schema migration.
- **Metrics** are registered via Hydra configs with `_target_` and `_partial_: True`. Parameters are bound at config time, not at call time, so the evaluation engine doesn't need to know what parameters each metric takes.
- **Algorithms** are either `LatentModule` (fit/transform) or `LightningModule` subclasses (training loops). The execution engine dispatches on type, not on name.

The result is a system where you can add a new algorithm, metric, dataset, or entire domain extension without touching core code.

## Two Execution Modes

manyLatents provides two ways to run:

**CLI** (`python -m manylatents.main`) executes a single step: one algorithm + metrics on one dataset. This is the primary user-facing interface and what SLURM jobs invoke.

**Python API** (`manylatents.api.run()`) is the programmatic interface designed for agent-driven workflows. It accepts `input_data` to chain the output of one call into the next, and supports `pipeline` configs for sequential steps within a single process. External orchestrators (like manyAgents) call this API directly to compose multi-step workflows without subprocess overhead.

```python
from manylatents.api import run

# Single step
result = run(
    data='swissroll',
    algorithms={'latent': {
        '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
        'n_components': 50
    }}
)

# Chaining: feed output of one step into the next
result2 = run(
    input_data=result['embeddings'],
    algorithms={'latent': {
        '_target_': 'manylatents.algorithms.latent.phate.PHATEModule',
        'n_components': 2
    }}
)
```

## Namespace Extension via `pkgutil`

manyLatents uses Python's namespace package mechanism to allow extensions to add modules under the `manylatents` namespace without forking or monkey-patching.

The core package's `__init__.py` contains one line:

```python
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

This tells Python: "if another installed package also defines a `manylatents` directory, merge its contents into mine." An extension like `manylatents-omics` can then provide `manylatents/dogma/`, `manylatents/popgen/`, and `manylatents/singlecell/` as top-level submodules.

The rule is simple: **core never imports from extensions; extensions import from core.**

### Hydra Config Discovery

Extensions also need their Hydra configs discovered. Each extension registers a `SearchPathPlugin`:

```python
class OmicsSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        search_path.append(
            provider="manylatents-omics",
            path="pkg://manylatents.dogma.configs",
        )
```

This is registered as a Hydra plugin entry point in `pyproject.toml`, so Hydra discovers extension configs automatically when the package is installed.

## Four Extension Axes

### 1. Algorithms

There are exactly two base classes, and the decision rule is binary:

- **`LatentModule`** — for algorithms that don't train with backprop. Subclass it, implement `fit(x, y=None)` and `transform(x)`, and you're done. PCA, UMAP, t-SNE, PHATE, DiffusionMap, MDS, and Archetypes all use this. Frozen foundation models also use this — the **FoundationEncoder pattern** is a LatentModule where `fit()` is a no-op and `transform()` wraps a pretrained model. It is a usage convention, not a separate class.

- **`LightningModule` subclasses** — for trainable neural networks. Implement `setup()`, `training_step()`, `encode()`, and `configure_optimizers()`. Autoencoders, VAEs, and Latent ODEs use this.

```python
# Adding a new LatentModule algorithm:
from manylatents.algorithms.latent.latent_module_base import LatentModule

class MyAlgorithm(LatentModule):
    def __init__(self, n_components=2, my_param=1.0, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.my_param = my_param

    def fit(self, x, y=None):
        # Fit your model
        self._is_fitted = True

    def transform(self, x):
        # Return reduced representation
        return x[:, :self.n_components]  # placeholder
```

Optional methods `kernel_matrix()` and `affinity_matrix()` can be implemented if the algorithm uses a kernel-based approach — this enables module-level metrics like `KernelMatrixSparsity` and `AffinitySpectrum`.

### 2. Metrics

Metrics follow the `Metric` protocol:

```python
def __call__(
    self,
    embeddings: np.ndarray,
    dataset=None,
    module=None,
    _knn_cache=None,
) -> float | tuple[float, np.ndarray] | dict[str, Any]
```

Three evaluation contexts determine which arguments are populated:

| Context | `embeddings` | `dataset` | `module` | Use case |
|---------|-------------|-----------|----------|----------|
| `embedding` | Low-dim output | Source dataset | - | Trustworthiness, continuity, kNN preservation |
| `dataset` | - | Source dataset | - | Stratification, admixture scores |
| `module` | - | Source dataset | Fitted LatentModule | Affinity spectrum, kernel sparsity |

Parameters like `n_neighbors`, `k`, or `return_per_sample` are set in the Hydra config with `_partial_: True`, not at call time. This means the evaluation engine can run any metric without knowing its parameters.

List-valued parameters in configs expand via Cartesian product through `flatten_and_unroll_metrics()`. For example, `n_neighbors: [5, 10, 20]` produces three separate metric evaluations.

Metrics that need kNN graphs can accept `_knn_cache` — a shared cache computed once with `max(k)` across all metrics that need it.

### 3. Data Modules

Data modules provide the `get_data()` method that returns the data matrix. They are auto-discovered at import time through the registry in `manylatents/data/__init__.py`.

Synthetic datasets (SwissRoll, Torus, SaddleSurface, GaussianBlobs, DLATree) generate data on-the-fly and work offline. File-based datasets (PrecomputedDataModule) load from disk.

For LightningModule algorithms, data modules also implement the `LightningDataModule` interface (`train_dataloader()`, etc.).

### 4. Domain Extensions

A domain extension is a separate installable package that adds algorithms, metrics, and data modules to the `manylatents` namespace. The reference implementation is `manylatents-omics`, which adds:

- `manylatents.dogma` — Foundation model encoders (Evo2, ESM3, Orthrus, AlphaGenome) and fusion algorithms
- `manylatents.popgen` — Population genetics data modules and domain-specific metrics (GeographicPreservation, AdmixturePreservation)
- `manylatents.singlecell` — Single-cell AnnData data modules

To create a new domain extension:

1. Create a package with `manylatents/` as its top-level directory
2. Add `__path__ = __import__('pkgutil').extend_path(__path__, __name__)` to `__init__.py`
3. Register a `SearchPathPlugin` for Hydra config discovery
4. Import from core (`from manylatents.algorithms.latent.latent_module_base import LatentModule`), never the other way around

## Hydra Configuration

Every extensible component has a corresponding Hydra config group:

```
configs/
  algorithms/
    latent/         # LatentModule configs (pca, umap, tsne, phate, ...)
    lightning/      # LightningModule configs (ae_reconstruction, latent_ode, ...)
      loss/         # Loss function configs
      network/      # Network architecture configs
      optimizer/    # Optimizer configs
  data/             # Dataset configs
  metrics/
    embedding/      # Embedding-level metric configs
    dataset/        # Dataset-level metric configs
    module/         # Module-level metric configs
    sampling/       # Metric sampling strategies
  callbacks/embedding/  # Embedding callback configs
  experiment/       # Experiment preset configs
  trainer/          # Lightning trainer configs
  logger/           # Logger configs (none, wandb)
  cluster/          # SLURM cluster configs (via Shop)
  launcher/         # Job launcher configs (via Shop)
```

A minimal config for a new algorithm:

```yaml
# configs/algorithms/latent/my_algo.yaml
_target_: manylatents.algorithms.latent.my_algo.MyAlgorithm
n_components: 2
my_param: 1.0
```

A minimal config for a new metric:

```yaml
# configs/metrics/embedding/my_metric.yaml
_target_: manylatents.metrics.my_metric.MyMetric
_partial_: true
n_neighbors: 5
```

The `_partial_: True` flag is critical for metrics — it tells Hydra to create a partially-applied callable rather than instantiating immediately. The evaluation engine then calls it with the runtime arguments (`embeddings`, `dataset`, etc.).

## Scope Boundaries

manyLatents owns single-step execution and the Python API for composable workflows. It does NOT own:

- **Multi-step orchestration logic** — that's manyAgents, which calls `manylatents.api.run()` to compose steps
- **Reinforcement learning / reward-driven training** — that's Geomancer
- **Cluster job submission** — that's Shop, which provides Hydra launcher plugins

These boundaries are by design. manyLatents stays focused on doing one thing well — running an algorithm with metrics on a dataset — and exposes clean interfaces for higher-level systems to compose.

## Adding a Complete Extension: Walkthrough

Here's how you'd add a new domain extension for, say, imaging data:

```
manylatents-imaging/
  manylatents/
    __init__.py              # extend_path line
    imaging/
      __init__.py
      data/
        __init__.py
        microscopy.py        # MicroscopyDataModule
      metrics/
        __init__.py
        ssim.py              # Structural similarity metric
      configs/
        data/
          microscopy.yaml
        metrics/
          embedding/
            ssim.yaml
      plugins/
        search_path.py       # ImagingSearchPathPlugin
  pyproject.toml             # Entry point for Hydra plugin
```

After `uv add ./manylatents-imaging`, users can run:

```bash
python -m manylatents.main algorithms/latent=pca data=microscopy metrics/embedding=ssim
```

No changes to core required.

## Release Readiness

The extension architecture is stable and in use by `manylatents-omics`. The following are known areas for future work:

- **Config group rename**: `algorithms/latent/` should become `algorithms/latent_module/` to better distinguish from `algorithms/lightning/`
- **Per-metric sampling**: Allow different sampling strategies per metric via a `_sampling` key in metric configs
- **Structured configs**: Migration from dict-based to dataclass-based Hydra configs for better validation
