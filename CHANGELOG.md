# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `input_metric` (unified, like `neighborhood_size`; top-level Hydra key `input_metric=...`): routed to UMAP
  `metric`, t-SNE `metric`, PHATE `knn_dist` and Isomap `metric`. `"precomputed"` takes an `(n, n)` distance matrix.
  `None` keeps each module's default. Other modules raise for a non-Euclidean value instead of ignoring it. t-SNE
  raises for a PCA start on a precomputed input.
- `PHATEModule(knn_dist=...)`: passed to phate. `"precomputed_distance"` or `"precomputed_affinity"` take an
  `(n, n)` matrix as input and turn off the PCA step. The default `"euclidean"` keeps prior behavior.
- `IsomapModule` (`algorithms/latent=isomap`, `run(algorithm="isomap")`): Isomap through
  `sklearn.manifold.Isomap`. `neighborhood_size` maps to sklearn `n_neighbors`.
- `LocallyLinearModule` (`algorithms/latent=lle`, `run(algorithm="lle")`): locally linear
  embedding through `sklearn.manifold.LocallyLinearEmbedding`, with `method` in
  `standard`, `hessian`, `modified` and `ltsa`. `neighborhood_size` maps to sklearn
  `n_neighbors`, and `random_state` reaches the sklearn eigen solver.
- `UMAPModule(init=...)`: a string or an `(n, n_components)` array, passed to umap-learn. The default
  `'spectral'` keeps prior behavior. The torchdr backend raises a `ValueError` for any other value.
- `UMAPModule(repulsion_strength=...)`: passed to umap-learn; `None` keeps the umap-learn default (1.0).
- `TSNEModule(late_exaggeration=..., dof=...)`: exaggeration in the late openTSNE phase (`None` = 1) and the
  degrees of freedom of the output kernel (1 = Cauchy, smaller = heavier tails), applied to both phases.
  Defaults reproduce the earlier layout. The torchdr backends raise if these are set.

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
