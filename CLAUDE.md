# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
uv sync                    # Install dependencies and create virtual environment
source .venv/bin/activate  # Activate virtual environment
```

### Testing
```bash
pytest                                           # Run all tests
pytest manylatents/algorithms/test_latentmodule_hydra.py # Run specific test file
pytest manylatents/tests/algorithms/dr_compliance_test.py # Run compliance tests for DR modules
```

### Running Experiments
```bash
python -m manylatents.main experiment=hgdp_pca                                    # Basic experiment
python -m manylatents.main experiment=hgdp_pca algorithm.dimensionality_reduction.n_components=10  # Override hyperparameters
python -m manylatents.main experiment=multiple_algorithms                        # Sequential pipeline experiment
```

### Code Quality
```bash
pre-commit run --all-files  # Run linting and formatting
```

## Architecture Overview

### Core Framework Structure
The codebase is built around **PyTorch Lightning**, **Hydra**, and **uv** for dependency management. The system provides a unified interface for applying dimensionality reduction and neural network techniques to diverse datasets.

### Key Components

#### 1. Latent Module System (`manylatents/algorithms/`)
- **Base Class**: `LatentModule` in `latent_module_base.py` provides the core interface with `fit()`, `transform()`, and `fit_transform()` methods
- **Algorithms**: Traditional DR methods (PCA, t-SNE, PHATE, UMAP) and neural network modules all inherit from this base
- **Neural Networks**: Located in `manylatents/algorithms/networks/` - autoencoder and other architectures that can be used standalone or chained with DR methods

#### 2. Unified Experiment Pipeline (`manylatents/main.py`)
The main entry point supports two execution modes and two types of algorithms:

**Execution Modes:**
- **Single Algorithm Mode**: Run one algorithm (backward compatible)
- **Sequential Pipeline Mode**: Chain multiple algorithms in sequence

**Algorithm Types:**
- **LatentModule instances**: Traditional DR algorithms that implement fit/transform interface
- **LightningModule instances**: Neural network models that can be trained with PyTorch Lightning

Key pipeline stages:
1. Data instantiation and loading
2. Algorithm instantiation (can be multiple algorithms in sequence)
3. Latent embedding computation (fit/transform for traditional methods, training for neural networks)
4. Evaluation with configurable metrics
5. Callback processing for visualization and logging

**Sequential Pipeline Features:**
- Algorithms run in order, with each algorithm receiving the output of the previous
- Supports mixed algorithm types (e.g., PCA → Autoencoder → final embedding)
- Per-step hyperparameter overrides via YAML configuration
- Full interpolation support for dynamic configuration (e.g., `${data}` references)

#### 3. Configuration System (`manylatents/configs/`)
Highly modular Hydra-based configuration:
- **Experiments**: Pre-defined combinations in `manylatents/configs/experiment/`
- **Algorithms**: Located in `manylatents/configs/algorithms/` with separate configs for DR methods (`latent/`) and neural models (`lightning/`)
- **Data**: Dataset-specific configurations
- **Metrics**: Configurable evaluation metrics for embeddings and models
- **Sweeps**: Hyperparameter sweep configurations for large-scale experiments

**Pipeline Configuration Format:**
```yaml
# Sequential pipeline example
pipeline:
  - latent/pca                              # Simple algorithm reference
  - lightning/ae_reconstruction:            # Algorithm with parameter overrides
      network:
        latent_dim: 2
        input_dim: 2
```

#### 4. Data Management (`manylatents/data/`)
- Supports various dataset types: genomic (HGDP, AOU, UKBB), synthetic (Swiss roll, saddle surface), and single-cell data
- **Precomputed Mixin**: Allows loading pre-computed embeddings for evaluation-only runs
- **Split vs Full modes**: Configurable train/test splitting

#### 5. Evaluation System (`manylatents/metrics/`)
Comprehensive metric computation using single dispatch pattern:
- **Embedding metrics**: Trustworthiness, continuity, participation ratio, fractal dimension, etc.
- **Dataset-specific metrics**: Geographic preservation, admixture preservation for genomic data
- **Model metrics**: For neural network evaluation

#### 6. Callback System
- **Lightning Callbacks**: Standard PyTorch Lightning callbacks for training
- **Embedding Callbacks**: Custom callbacks for post-processing embeddings (plotting, saving, logging to W&B)

### Adding New Components

#### New Dimensionality Reduction Method
1. Create class inheriting from `LatentModule` in `manylatents/algorithms/yourmethod.py`
2. Add Hydra config in `manylatents/configs/algorithm/dimensionality_reduction/yourmethod.yaml`
3. Write tests in `manylatents/algorithms/yourmethod_test.py`
4. Run compliance test to ensure interface compatibility

#### New Neural Network Architecture
1. Create class inheriting from `LightningModule` in `manylatents/algorithms/networks/yournet.py`
2. Add configs in `manylatents/configs/algorithm/model/network/yournet.yaml`
3. Configure loss functions and optimizers as needed
4. Test with the unified pipeline through `manylatents/main.py`

### Important Notes
- **Sequential Pipeline Support**: The system supports chaining algorithms (e.g., PCA → Autoencoder → final embedding)
- **Algorithm Resolution**: Uses `manylatents/utils/pipeline.py` to resolve pipeline configurations and handle interpolations
- **Hydra Integration**: All algorithms are instantiated via Hydra, enabling easy configuration and hyperparameter sweeps
- **Evaluation System**: The `evaluate()` function uses single dispatch to handle both embedding dictionaries and LightningModule instances
- **Output Management**: Results are saved in `outputs/<date>/<time>/` with full experiment logging and W&B integration

### API Metrics Configuration

**IMPORTANT**: When using `manylatents.api.run()` directly, metric configurations MUST include `_target_` keys pointing to the metric class. Empty dicts `{}` will NOT work.

**This will NOT compute metrics (empty scores):**
```python
result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': '...PCA', 'n_components': 10}},
    metrics={
        'embedding': {
            'trustworthiness': {},  # Missing _target_ - WILL BE SKIPPED
            'continuity': {}        # Missing _target_ - WILL BE SKIPPED
        }
    }
)
# result['scores'] == {}  (empty!)
```

**Correct usage with full metric configs:**
```python
result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': '...PCA', 'n_components': 10}},
    metrics={
        'embedding': {
            'trustworthiness': {
                '_target_': 'manylatents.metrics.trustworthiness.Trustworthiness',
                '_partial_': True,
                'n_neighbors': 5,
                'metric': 'euclidean'
            },
            'continuity': {
                '_target_': 'manylatents.metrics.continuity.Continuity',
                '_partial_': True,
                'return_per_sample': True
            },
            'local_intrinsic_dimensionality': {
                '_target_': 'manylatents.metrics.lid.LocalIntrinsicDimensionality',
                '_partial_': True,
                'k': 20
            }
        }
    }
)
# result['scores'] contains computed metrics
```

**Why this matters:**
- The `flatten_and_unroll_metrics()` function in `manylatents/utils/metrics.py` filters for configs with `_target_` keys
- The manyAgents adapter handles this transformation automatically by loading configs from `manylatents/configs/metrics/`
- When calling the API directly, you must provide the full Hydra-style configs

**Available metric configs** (in `manylatents/configs/metrics/embedding/`):
- `trustworthiness.yaml` - Local neighborhood preservation
- `continuity.yaml` - Reverse trustworthiness
- `local_intrinsic_dimensionality.yaml` - Local intrinsic dimensionality (LID)
- `participation_ratio.yaml` - Effective dimensionality
- `fractal_dimension.yaml` - Fractal structure measure
- `knn_preservation.yaml` - k-NN graph preservation

### Metric Evaluation Architecture

**Entrypoints:**
- `manylatents/main.py` - Hydra CLI with smart routing (single vs pipeline)
- `manylatents/api.py` - Programmatic Python interface for manyAgents
- `manylatents/experiment.py` - Core execution (`run_algorithm`, `run_pipeline`)

**Hydra Partial Instantiation:**
All metrics use `_partial_: True` to create `functools.partial` objects. This defers parameter binding until call-time:

```yaml
trustworthiness:
  _target_: manylatents.metrics.trustworthiness.Trustworthiness
  _partial_: True      # Creates functools.partial
  n_neighbors: 5
```

At execution time (experiment.py:187-193):
```python
metric_fn = hydra.utils.instantiate(metric_cfg)  # → functools.partial
result = metric_fn(embeddings=emb, dataset=ds, module=module)  # Bound at call-time
```

**Multi-Scale Expansion:**
List-valued parameters trigger Cartesian product expansion via `flatten_and_unroll_metrics()`:

```yaml
n_neighbors: [15, 25, 50, 100, 250]  # Expands to 5 separate metrics
```

Naming convention: `embedding.trustworthiness__n_neighbors_15`, `embedding.trustworthiness__n_neighbors_25`, etc.

**Subsampling (DEPRECATED):**
```yaml
# OLD - deprecated, use sampling strategies instead
metrics:
  subsample_fraction: 0.1
```

### Sampling Strategies (Implemented)

Pluggable sampling strategies for metric evaluation. Located in `manylatents/utils/sampling.py`.

**Available Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `RandomSampling` | Random without replacement | Default, fast |
| `StratifiedSampling` | Preserves label distribution | Population-balanced metrics |
| `FarthestPointSampling` | Maximizes embedding coverage | Representative subsets |
| `FixedIndexSampling` | Uses precomputed indices | Cross-setting comparisons |

**Usage via Hydra Config:**
```yaml
metrics:
  sampling:
    _target_: manylatents.utils.sampling.RandomSampling
    seed: 42
    fraction: 0.1
  embedding:
    trustworthiness: {}
```

**Deterministic Index Workflow (for comparing across settings):**
```python
from manylatents.utils.sampling import RandomSampling, FixedIndexSampling

# 1. Precompute indices once
sampler = RandomSampling(seed=42)
indices = sampler.get_indices(n_total=1000, fraction=0.1)
np.save('shared_indices.npy', indices)

# 2. Use same indices across different algorithm settings
fixed = FixedIndexSampling(indices=np.load('shared_indices.npy'))
emb_sub_A, ds_sub_A, _ = fixed.sample(embeddings_A, dataset_A)
emb_sub_B, ds_sub_B, _ = fixed.sample(embeddings_B, dataset_B)
```

**Config Files:**
- `configs/metrics/sampling/random.yaml`
- `configs/metrics/sampling/stratified.yaml`
- `configs/metrics/sampling/farthest_point.yaml`

### Sampling Roadmap (Future)

**Phase 2: Per-Metric Sampling** (Planned)
```yaml
metrics:
  embedding:
    trustworthiness:
      _sampling:
        strategy: stratified
        fraction: 0.1
        stratify_by: population_label
```

**Phase 3: Adaptive Sampling** (Planned)
Automatic sampling based on dataset size and metric complexity (O(n²) metrics get smaller samples).

**Phase 4: Reproducible Sampling with Caching** (Planned)
Cache sampled subsets for reproducibility across runs.

### Hydra `null` Override Issue

**IMPORTANT**: Hydra does NOT support `null` as an override value. This causes `ValueError: Config group override must be a string or a list. Got NoneType`.

**Problem Examples (these FAIL):**
```bash
python -m manylatents.main experiment=single_algorithm callbacks=null
python -m manylatents.main experiment=single_algorithm +callbacks=null
python -m manylatents.main experiment=single_algorithm callbacks/embedding=null
```

**Why this matters:**
- Cannot disable config groups via command-line overrides
- Cannot pass `null` through manyAgents adapter to manyLatents
- Forces us to create explicit "none" or "minimal" config files

**Current Workarounds:**
1. Use `debug=true` to set wandb mode to "disabled" (but still initializes wandb with overhead)
2. Create explicit config files like `metrics/null.yaml` with `defaults: [dataset: null, embedding: null, module: null]`
3. Design default configs to be minimal/fast, with separate `_with_wandb` variants for logging

**Solution (IMPLEMENTED):**
A `logger` config group now controls experiment-level wandb logging:
- `logger=none` - No wandb initialization, fastest (default for CI/testing)
- `logger=wandb` - Full wandb integration

This allows `python -m manylatents.main experiment=test logger=none` to cleanly disable logging without fighting Hydra's type system or requiring environment variables.

**Usage:**
```bash
# No logging (CI/testing)
python -m manylatents.main experiment=single_algorithm logger=none

# Full wandb logging
python -m manylatents.main experiment=single_algorithm logger=wandb
```

### Pipeline Examples

**Basic Sequential Pipeline:**
```yaml
# experiment/my_pipeline.yaml
pipeline:
  - latent/pca
  - latent/umap
```

**Complex Mixed Pipeline:**
```yaml
# experiment/complex_pipeline.yaml  
pipeline:
  - latent/pca:
      n_components: 50
  - lightning/ae_reconstruction:
      network:
        input_dim: 50
        latent_dim: 2
        hidden_dims: [128, 64]
      optimizer:
        lr: 0.01
```