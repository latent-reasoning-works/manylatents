# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

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
