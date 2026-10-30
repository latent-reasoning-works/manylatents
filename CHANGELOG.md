# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Relicensed from MIT to Apache-2.0. Releases up to and including 0.1.8 stay under MIT; NOTICE keeps the MIT notice and lists the MIOFlow files it does not cover.


### Added
- `THIRD_PARTY_NOTICES.md`: the nine files ported from KrishnaswamyLab code (seven modules
  and the two MIOFlow configs), listed under their upstream repositories (MIOFlow, cflows,
  AAnet) with the verbatim Yale non-commercial license text and the upstream commit it was
  copied from. A separate `archetypes` section carries the BSD-3-Clause notice for the two
  functions that `aa.py` adapts. Each of the nine files now starts with
  `# SPDX-License-Identifier: LicenseRef-Yale-Non-Commercial`. `NOTICE` lists all nine files
  and names the three upstream repositories. Nothing moves: import paths, extras and the
  package license metadata are unchanged.
- `ReebGraphModule(max_N=5000)`: a cell cap. Above `max_N` rows, the Reeb graph is built on a
  `max_N`-row sample drawn with the `PersistentHomology` procedure, seeded by `random_state`.
  Rows that were not sampled take the membership of their nearest sampled row, so the
  embedding keeps one row per input row. `extra_outputs()` gains `subsample_indices`. The four
  `reeb_graph*` configs set `max_N: 5000`.
- Generic `graph` output (`manylatents.outputs.graph`): `edges`, `edge_weights`, `node_ids` and
  `node_coordinates` for any algorithm with `adjacency()` and `node_coordinates`.
- `SaveOutputs` with `save_format: csv` writes the `graph` output as `<base>_nodes.csv`,
  `<base>_edges.csv` and, with manykinds installed, a `manykinds.SparseGraph` archive
  `<base>_graph.npz`. The paths are returned as `graph_nodes_path`, `graph_edges_path` and
  `graph_npz_path`.
- `manylatents.utils.charts.ball_cover(X, m, cache=None)`: covers `X` with kNN balls of size `m`
  (the landmark counts as a member) around landmarks chosen by farthest-point sampling from row 0. Returns a
  `BallCover(landmarks, members)` NamedTuple with a 2-D member array. It is deterministic. Its `cache` goes to
  `compute_knn`, so one dict primed at the largest `m` serves every smaller `m`.
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
- `UMAPModule(init=...)`: a string or an `(n, n_components)` array, passed to umap-learn. The default
  `'spectral'` keeps prior behavior. The torchdr backend raises a `ValueError` for any other value.

### Changed
- A Reeb graph run on more than 5000 rows now uses a 5000-row sample by default (it used all
  rows, or ran out of memory). Pass `max_N=None` for the old behaviour.

### Fixed
- `ArchetypalAnalysisModule` (`algorithms/latent=aa`) fits again under the locked archetypes 0.6.2,
  scikit-learn 1.8 and NumPy 2. It passed the archetypes 0.11 keywords `method` and `method_kwargs`,
  so every fit raised `TypeError`. It now passes `n_archetypes`, `max_iter` and `random_state` to a
  private subclass of `archetypes.AA` that restores `_validate_data` (removed in scikit-learn 1.6)
  and avoids the `np.mat` call (removed in NumPy 2.0) in the furthest-sum start. `method` is still
  accepted and is not forwarded.
- `ArchetypalAnalysisModule` no longer raises `RuntimeError: Maximum number of iterations reached.`
  from scipy `nnls` (for example `data=swissroll` with 3 archetypes on float32 input, or 8
  archetypes on any input). It fits in float64, and its solver sends each wide subproblem (more
  unknowns than equations, as in the archetype step) to `scipy.optimize.lsq_linear(method="bvls")`,
  which solves the same problem; other subproblems use `nnls`, falling back to `bvls` at its
  iteration cap. Where plain `nnls` converged, the fit is unchanged to within round-off.

## [0.1.8] - 2026-10-01

### Added
- `grouped_average_precision_difference`: paired score comparison with row and optional cluster bootstrap confidence intervals.
- `gpd_lid` metric and `manylatents.metrics.gpd_lid`: Hill and Pickands tail-index estimators, within one cloud or against a separate reference cloud.
- `compute_knn_query`: neighbours of query points in a separate reference set.
- `manylatents.metrics.reference_geometry`: kNN distance, LOF novelty, PCA split, Mahalanobis distance, local participation ratio and standardization against a reference cloud.
- `manylatents.utils.surrogates`: row shuffle, random feature subset, covariance-matched Gaussian and within-group permutation.
- `grouped_average_precision` (bootstrap over rows or over whole groups) and `partial_spearman`.
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
