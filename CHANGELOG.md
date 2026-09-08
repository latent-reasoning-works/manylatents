# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIOFlow: optional composed GAGA (Geometry-Aware Generative Autoencoder)
  encoder (`use_gaga=True`). When enabled, a GAGA encoder/decoder is
  pretrained internally during `setup()` (two-phase: distance-preserving
  encoder against PHATE-embedding distances, then reconstruction-only
  decoder), then frozen, and the Neural ODE flow trains/integrates through
  its latent space instead of raw ambient space. Unlike upstream `mioflow`
  2.0 (which keeps GAGA and MIOFlow as two classes the caller manually
  composes), manylatents composes them into a single `MIOFlow`
  LightningModule with one Hydra catalog entry. See issue #293.
- MIOFlow: `_group_by_time` now accepts `batch["time"]` as a third alias
  alongside `"label"`/`"labels"`, matching the key `manylatents.api.run()`'s
  generic pipeline emits.
- MIOFlow: `encode()` accepts optional explicit `t_start`/`t_end` instead of
  always inferring the integration span from the datamodule.
- MIOFlow: opt-in `grad_clip` (default `None`, preserving prior behavior;
  upstream `mioflow` 2.0 defaults this to `1.0`).
- MIOFlow: `density_top_k`/`density_hinge_value` are now real constructor
  kwargs instead of hardcoded values in the density loss call sites.
- `MIOFlowODEFunc`: optional `momentum_beta` velocity smoothing (default
  `0.0`, matching upstream `mioflow` 2.0's `ODEFunc.momentum_beta`).
- New config: `algorithms/lightning/mioflow_gaga.yaml`, a demo variant with
  `use_gaga: true` preset.

## [0.1.6] - 2026-06-03

### Fixed
- `compute_geodesic_distances`: cast KNN graph index arrays to int32 before
  `scipy.sparse.csgraph.shortest_path`, which raised "Buffer dtype mismatch,
  expected 'const int' but got 'long'" on numpy>=2 / scipy 1.14. Unblocks
  manylatents-omics' `AdmixturePreservation` metric.

## [0.1.2] - 2026-02-27

### Removed
- GAGA and MIOFlow algorithms (source, configs, tests) — moved to `dev` branch for further testing

### Fixed
- Trainer default config: `gradient_clip_val: null` instead of `1.0`

### Changed
- Absorbed lightweight extras into core dependencies: `torchdiffeq`, `torchsde`, `POT`, `ripser`, `hydra-submitit-launcher`, `leidenalg`, `python-igraph`
- Renamed `mioflow-jax` extra to `jax` (bare `jax`, no `[cpu]` pin)
- Removed dead extras: `tracking`, `dynamics`, `transport`, `topology`, `cluster`, `clustering`, `mioflow`
- `all` extra now installs `manylatents[hf,torchdr,jax]`

## [0.1.0] - 2026-02-17

### Added
- Unified dimensionality reduction framework with Hydra config system
- 10 LatentModule algorithms: PCA, UMAP, t-SNE, PHATE, DiffusionMap, MDS, Archetypes, Multiscale PHATE, Classifier, NoOp
- 4 LightningModule algorithms: Autoencoder, AANet, Latent ODE, HuggingFace Trainer
- 30+ embedding/dataset/module metrics with decorator-based registry
- Python API via `manylatents.api.run()` with pipeline chaining
- Extension system via entry-point plugin discovery (`manylatents.plugins`)
- Shared kNN/SVD/eigenvalue cache infrastructure for metric computation
- Pluggable sampling strategies (random, stratified)
- Embedding callbacks: save, plot, wandb logging, loadings analysis
- CLI entry point: `manylatents` / `python -m manylatents`
- SLURM submission via `hydra-submitit-launcher`
- Optional GPU-accelerated DR via TorchDR backend
- MIT license
