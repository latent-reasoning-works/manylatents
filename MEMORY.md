# MEMORY.md

Semantic reference for agents working on manyLatents. Organized by topic, not chronology.

## Current State

- **Package version**: 0.1.0
- **Python**: >=3.11, <3.13
- **Framework**: PyTorch Lightning + Hydra + uv
- **Tests**: `tests/` contains tests for lightning adapters, latent ODE, loadings analysis, plot embeddings, and metric registry. No comprehensive module instantiation sweep test exists yet (see Pending).
- **Synthetic datasets**: SwissRoll, Torus, SaddleSurface, GaussianBlobs, DLATree work offline (generated on-the-fly)
- **External datasets**: PrecomputedDataModule requires file paths; omics datasets require external files
- **Verified CLI path**: `python -m manylatents.main algorithms/latent=pca data=swissroll` (LatentModule path)
- **LightningModule CLI**: NOT yet tested in CI (see TODO.md). Should work with `trainer.fast_dev_run=true`.

## Available Algorithms

### LatentModule Algorithms (fit/transform)

| Algorithm | Class | File | Config | Key Parameters |
|-----------|-------|------|--------|----------------|
| PCA | `PCAModule` | `algorithms/latent/pca.py` | `algorithms/latent=pca` | `n_components`, `random_state` |
| t-SNE | `TSNEModule` | `algorithms/latent/tsne.py` | `algorithms/latent=tsne` | `n_components`, `perplexity`, `learning_rate`, `metric` |
| UMAP | `UMAPModule` | `algorithms/latent/umap.py` | `algorithms/latent=umap` | `n_components`, `n_neighbors`, `min_dist`, `metric` |
| PHATE | `PHATEModule` | `algorithms/latent/phate.py` | `algorithms/latent=phate` | `n_components`, `knn`, `gamma` |
| MultiscalePHATE | `MultiscalePHATEModule` | `algorithms/latent/multiscale_phate.py` | `algorithms/latent=multiscale_phate` | `n_components`, `knn` |
| DiffusionMap | `DiffusionMapModule` | `algorithms/latent/diffusion_map.py` | `algorithms/latent=diffusionmap` | `n_components`, `knn`, `decay` |
| MDS | `MDSModule` | `algorithms/latent/multi_dimensional_scaling.py` | `algorithms/latent=mds` | `n_components` |
| Archetypes | `AAModule` | `algorithms/latent/aa.py` | `algorithms/latent=aa` | `n_components` |
| Classifier | `ClassifierModule` | `algorithms/latent/classifier.py` | `algorithms/latent=classifier` | Supervised; uses labels from `get_labels()` |
| Noop | `DRNoop` | `algorithms/latent/dr_noop.py` | `algorithms/latent=noop` | Passthrough (identity) |
| MergingModule | `MergingModule` | `algorithms/latent/merging.py` | `algorithms/latent=merging` | `strategy` (concat, weighted_sum, mean, concat_pca, svd), `target_dim` |

**FoundationEncoder pattern**: LatentModules where `fit()` is a no-op and `transform()` wraps a pretrained model. Implementations live in `manylatents-omics/manylatents/dogma/encoders/` (Evo2, ESM3, Orthrus, AlphaGenome). This is a usage convention within LatentModule, not a separate base class.

### LightningModule Algorithms (trainable)

| Algorithm | Class | File | Config | Key Parameters |
|-----------|-------|------|--------|----------------|
| Reconstruction | `Reconstruction` | `algorithms/lightning/reconstruction.py` | `algorithms/lightning=ae_reconstruction` | `network`, `optimizer`, `loss`, `init_seed` |
| AANet Reconstruction | (uses Reconstruction) | - | `algorithms/lightning=aanet_reconstruction` | Uses AANet network config |
| LatentODE | `LatentODE` | `algorithms/lightning/latent_ode.py` | `algorithms/lightning=latent_ode` | `network`, `optimizer`, `loss`, `integration_times` |
| HF Trainer | [UNVERIFIED] | - | `algorithms/lightning=hf_trainer` | HuggingFace model training |

### Networks (nn.Module, used by LightningModules)

| Network | Class | File | Config |
|---------|-------|------|--------|
| Autoencoder | `Autoencoder` | `algorithms/lightning/networks/autoencoder.py` | `algorithms/lightning/network=autoencoder` |
| AANet | `AANet` | `algorithms/lightning/networks/aanet.py` | `algorithms/lightning/network=aanet` |

### Losses

| Loss | File | Config |
|------|------|--------|
| MSELoss | `algorithms/lightning/losses/mse.py` | `algorithms/lightning/loss=default` |
| GeometricLoss | `algorithms/lightning/losses/geometric.py` | `algorithms/lightning/loss=ae_dim`, `ae_neighbors`, `ae_shape` |

## Available Metrics

### Embedding Metrics (evaluate low-dim embeddings)

| Metric | Class/Function | File | Config | Return Type |
|--------|---------------|------|--------|-------------|
| Trustworthiness | `Trustworthiness` | `metrics/trustworthiness.py` | `metrics/embedding=trustworthiness` | float; supports `_knn_cache` |
| Continuity | `Continuity` | `metrics/continuity.py` | `metrics/embedding=continuity` | float or (float, ndarray) with `return_per_sample` |
| ParticipationRatio | `ParticipationRatio` | `metrics/participation_ratio.py` | `metrics/embedding=participation_ratio` | float |
| FractalDimension | `FractalDimension` | `metrics/fractal_dimension.py` | `metrics/embedding=fractal_dimension` | float |
| KNNPreservation | `KNNPreservation` | `metrics/knn_preservation.py` | `metrics/embedding=knn_preservation` | float; supports `_knn_cache` |
| LocalIntrinsicDimensionality | `LocalIntrinsicDimensionality` | `metrics/lid.py` | `metrics/embedding=local_intrinsic_dimensionality` | float; param `k` |
| Anisotropy | `Anisotropy` | `metrics/anisotropy.py` | `metrics/embedding=anisotropy` | float |
| MagnitudeDimension | `MagnitudeDimension` | `metrics/magnitude_dimension.py` | `metrics/embedding=magnitude_dimension` | float |
| PersistentHomology | `PersistentHomology` | `metrics/persistent_homology.py` | `metrics/embedding=persistent_homology` | dict (beta_0, beta_1, etc.) |
| TangentSpaceApproximation | `TangentSpaceApproximation` | `metrics/tangent_space.py` | `metrics/embedding=tangent_space` | float |
| PearsonCorrelation | `PearsonCorrelation` | `metrics/correlation.py` | `metrics/embedding=pearson_correlation` | float |
| DiffusionCurvature | `DiffusionCurvature` | `metrics/diffusion_curvature.py` | `metrics/embedding=diffusion_curvature` | ndarray (per-sample curvature) |
| DiffusionSpectralEntropy | `DiffusionSpectralEntropy` | `metrics/diffusion_spectral_entropy.py` | - | float |
| AUC | `AUC` | `metrics/auc.py` | - | float (requires labels) |
| OutlierScore | `OutlierScore` | `metrics/outlier_score.py` | - | float |

### Dataset Metrics (evaluate on dataset-level info)

| Metric | Function | File | Config | Return Type |
|--------|----------|------|--------|-------------|
| Stratification | `kmeans_stratification` | `metrics/stratification.py` | `metrics/dataset=stratification` | dict |
| AdmixtureLaplacian | - | - | `metrics/dataset=admixture_laplacian` | float |
| SampleId | `SampleId` | - | `metrics/dataset=sample_id` | ndarray (sample IDs) |

### Module Metrics (evaluate on fitted LatentModule)

| Metric | Function | File | Config | Return Type |
|--------|----------|------|--------|-------------|
| AffinitySpectrum | `AffinitySpectrum` | `metrics/affinity_spectrum.py` | `metrics/module=affinity_spectrum` | ndarray (eigenvalues) |
| ConnectedComponents | `ConnectedComponents` | `metrics/connected_components.py` | `metrics/module=connected_components` | ndarray |
| KernelMatrixSparsity | `KernelMatrixSparsity` | `metrics/kernel_matrix_sparsity.py` | `metrics/module=kernel_matrix_sparsity` | float |
| KernelMatrixDensity | `KernelMatrixDensity` | `metrics/kernel_matrix_sparsity.py` | `metrics/module=kernel_matrix_density` | float |
| DiffusionMapCorrelation | `DiffusionMapCorrelation` | `metrics/diffusion_map_correlation.py` | `metrics/module=diffusion_map_correlation` | float |

### Cross-Modal Alignment Metrics

| Metric | Class | File | Return Type |
|--------|-------|------|-------------|
| CKA | `CKA`, `cka_pairwise` | `metrics/cka.py` | float (linear or RBF kernel) |
| CrossModalJaccard | `CrossModalJaccard`, `cross_modal_jaccard_pairwise` | `metrics/cross_modal_jaccard.py` | float |
| RankAgreement | `RankAgreement` | `metrics/rank_agreement.py` | float |
| AlignmentScore | `AlignmentScore` | `metrics/alignment_score.py` | float (with StratificationResult) |

### Metric Bundles (composite configs)

| Config | Metrics Included |
|--------|-----------------|
| `metrics=fidelity_metrics` | trustworthiness, continuity, knn_preservation |
| `metrics=manifold_suite` | participation_ratio, fractal_dimension, tangent_space, persistent_homology |
| `metrics=null` | No metrics |
| `metrics=test_metric` | Lightweight test metric |

## Available Data Modules

### Synthetic (offline, no external files)

| Dataset | Class | File | Config | Description |
|---------|-------|------|--------|-------------|
| SwissRoll | `SwissRollDataModule` | `data/swissroll.py` | `data=swissroll` | 3D swiss roll manifold |
| Torus | `TorusDataModule` | `data/torus.py` | `data=torus` | Torus surface |
| SaddleSurface | `SaddleSurfaceDataModule` | `data/saddlesurface.py` | `data=saddle_surface` | Saddle-shaped surface |
| GaussianBlobs | `GaussianBlobsDataModule` | `data/gaussian_blobs.py` | `data=gaussian_blobs` / `data=clusters` / `data=one_blob` | Clustered blobs |
| DLATree | `DLATreeDataModule` | `data/dlatree.py` | `data=dla_tree` | Diffusion-limited aggregation tree |
| TestData | `TestDataModule` | `data/test_data.py` | `data=test_data` | Minimal test data |

### File-Based (require external data)

| Dataset | Class | File | Config | Description |
|---------|-------|------|--------|-------------|
| Precomputed | `PrecomputedDataModule` | `data/precomputed_datamodule.py` | `data=precomputed` | Load embeddings from files (.csv, .npy, .pt, .h5) |
| MHI Split | - | - | `data=mhi_split` | MHI dataset with train/test split |

### Auto-Discovery Registry

Datasets are auto-discovered at import time. Use `from manylatents.data import get_dataset, list_datasets` for programmatic access.

## Namespace Extensions

### manylatents-omics (v0.2.0)

**Installation**: `uv add git+https://github.com/latent-reasoning-works/manylatents-omics.git`

**Adds three submodules:**
- `manylatents.dogma` — Foundation model encoders (Evo2, ESM3, Orthrus, AlphaGenome) and fusion algorithms
- `manylatents.popgen` — Population genetics data modules (ManifoldGeneticsDataModule) and metrics (GeographicPreservation, AdmixturePreservation)
- `manylatents.singlecell` — Single-cell AnnData modules (PBMC, Embryoid Body)

**Entry point**: `python -m manylatents.omics.main` (registers OmicsSearchPathPlugin before Hydra init)

**Hydra plugin**: `OmicsSearchPathPlugin` registered via entry point in pyproject.toml and runtime in `manylatents.dogma.__init__`

**Known integration issues:**
- Circular dependency between manylatents and manylatents-omics in pyproject.toml (dogma extra commented out)
- AlphaGenome requires JAX with CUDA 12 and `torch-jax-interop`
- dogma extra requires wheelnext uv for prebuilt CUDA wheels (transformer-engine, mamba-ssm, flash-attn)

## Shop Integration

**Package**: `shop` (private repo, optional dependency via `uv sync --extra slurm`)

**Available cluster configs** (in manylatents/configs/cluster/):
- `mila` — Mila cluster, local submission (submitit_slurm)
- `mila_remote` — Mila cluster, remote submission via SSH (RemoteSlurmLauncher)
- `narval` — DRAC Narval cluster

**Available resource templates** (in shop/config_templates/resources/):
- `cpu` — CPU-only jobs (4 cores, 16GB)
- `cpu_large` — Large CPU jobs (16 cores, 64GB)
- `gpu` — Single GPU jobs (1 GPU, 4 cores, 32GB)

**Composing**: `python -m manylatents.main -m cluster=mila resources=gpu algorithms/latent=umap data=swissroll`

## Known Gotchas

### FoundationEncoder is a LatentModule pattern
FoundationEncoder is a usage convention within LatentModule, not a separate class. Implementations live in `manylatents-omics/manylatents/dogma/encoders/`. The old `algorithms/encoder/` directory and dangling import have been removed.

### Hydra `null` override limitation
Hydra CLI does not support `null` as an override value. Workaround: use explicit null config files (`metrics=null`, `logger=none`).

### API metrics require `_target_`
When calling `manylatents.api.run()`, metric configs with empty dicts `{}` are silently skipped. Must include `_target_` and `_partial_: True`.

### scipy upper bound
`scipy>=1.8,<1.15` is pinned for archetypes/PHATE compatibility. Do not relax.

### Loss function convention
Use project's `MSELoss` from `manylatents.algorithms.lightning.losses.mse`, not `torch.nn.MSELoss`. Signature: `(outputs, targets, **kwargs)`.

### LightningModule test setup
Unit tests not using `trainer.fit()` must call `model.setup()` manually.

### save_hyperparameters
Always use `self.save_hyperparameters(ignore=["datamodule", "network", "loss"])`.

### GlobalHydra state
Multiple Hydra calls in one process require clearing GlobalHydra. The API does this internally.

## Pending / Not Yet Done

### Pre-Release Blockers
- **Lightning Module CLI test**: No CI test for the LightningModule path. Should test `algorithms/lightning=ae_reconstruction` with `trainer.fast_dev_run=true`. (TODO.md, priority: High)
- **Module instantiation sweep test**: No pytest that sweeps all algorithm modules for import/instantiation errors. (TODO.md, priority: High)
- ~~**Remove `algorithms/encoder/` directory**~~: DONE. Directory and dangling import removed.
- **Config group rename**: `algorithms/latent/` should be renamed to `algorithms/latent_module/` per TODO in `configs/algorithms/latent/__init__.py`.

### Future Architecture
- **Full non-Lightning inference mode**: FoundationEncoder currently uses a workaround (LatentModule with no-op fit). A dedicated `inference` algorithm mode could be cleaner. (TODO.md, priority: Low)
- **Per-metric sampling** (Phase 2): Allow different sampling strategies per metric via `_sampling` key.
- **Adaptive sampling** (Phase 3): Auto-select sample size based on dataset size and metric complexity.
- **Reproducible sampling with caching** (Phase 4): Cache sampled subsets.

### Dependencies
- `scprep` is listed as TODO to remove after issue #182 (plotting replacement)
- Circular dependency with manylatents-omics dogma extra (commented out in pyproject.toml)
