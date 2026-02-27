# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
