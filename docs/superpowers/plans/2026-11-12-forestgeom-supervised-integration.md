# ForestGeom and supervised representation learning in manylatents: audit and plan

**Status:** plan for discussion. No library code in this PR. Nothing below is decided until the open
questions in the last section are answered.

**Goal:** make supervised, forest-guided dimensionality reduction a first-class part of
manylatents. That means forest proximities from
[ForestGeom](https://github.com/JakeSRhodesLab/ForestGeom) as a geometry source, RF-PHATE and
RF-AE as algorithms, and the evaluation protocol of the RF-AE paper
([arXiv:2502.13257](https://arxiv.org/abs/2502.13257)) as registered metrics.

**What exists today:** one supervised module, `ClassifierModule`
(`manylatents/algorithms/latent/classifier.py`). It is a logistic regression whose
"embedding" is one column, P(y=1|x), scored by one metric, `AUC`. It is meant for binary
labels, but it does not enforce that: it returns probability column 1 for any label array.

Everything in the "measured" tables was run on 2026-11-12 against ForestGeom 0.4.0 (commit
`053ef74`) and manylatents `ad0e10e`, with scikit-learn 1.8.0.

**Revision of 2026-11-13.** An independent review (Codex, read-only, at `8b39420`) checked
every claim about the code and every task brief. Its confirmed findings are folded in:
corrections to the audit in section 2, the "Constraints found in review" list in section 3,
a new task T0.0, and rewritten briefs for T0.1 to T0.5.

---

## 1. Audit of ForestGeom

### What it gives us

`forestgeom.Proximity(forest, weight_scheme)` wraps a tree ensemble and represents its
proximity as sparse leaf-incidence maps, `P = Q Wᵀ`, without building a dense pairwise matrix.

| Piece | What manylatents would use it for |
|---|---|
| `query_map(X)` / `reference_map()` | Sparse supervised features `Q` (n × n_leaves). Linear methods on `Q` are kernel methods on `P`, and `query_map(X_new)` is a native out-of-sample map. |
| `training_proximity()` / `transform(X_new)` | Train–train and test–train proximity blocks: the input to RF-PHATE and to RF-AE. |
| Schemes `uniform`, `kerf`, `oob`, `gap`, `boosted` | `gap` (RF-GAP) is what RF-PHATE and RF-AE use. `uniform`, `kerf`, `boosted` are symmetric kernels. |
| Adapters | scikit-learn RF, ExtraTrees, RandomTreesEmbedding (unsupervised), GBT; XGBoost and LightGBM as an extra; custom forests through four methods. |
| `training_proximity(adjust_diagonal=True, force_symmetric=True)` | For `gap`, this is the redefined self-similarity and the symmetrisation of RF-AE §3 (to be confirmed, question 4). |

It installs and runs inside the manylatents environment (Python 3.11, scipy 1.14.1,
scikit-learn 1.8.0). Its own suite passes: 198 tests.

### What it does not give us

RF-PHATE, RF-AE, prototype selection (class-wise k-medoids), row normalisation of
proximities, and the evaluation metrics are not in the package. `experiments/` has k-NN test
accuracy and a PageRank PHATE variant as scripts, not as importable code.

### Defects found (all reproduced)

| # | Finding | Measured | Consequence for us |
|---|---|---|---|
| F1 | `get_oob_mask` and `get_in_bag_counts` (`adapters/rf_et_rte.py`) rebuild the bootstrap with `n_samples_bootstrap = n_samples`. They ignore the forest's `max_samples`. | `max_samples=0.3`, 50 trees: 37.0% of out-of-bag entries and 50.2% of in-bag counts differ from `forest.estimators_samples_`. With `max_samples=None`: 0%. No error, and GAP rows still sum to 1. | `oob` and `gap` are silently wrong for any subsampled forest. Subsampling is the usual way to make large forests affordable. The manylatents wrapper must refuse `max_samples` with these schemes until this is fixed upstream. |
| F2 | A fitted forest is reused as-is, and `X` passed to `Proximity.fit` is assumed to be its training set. Nothing checks this. | Forest trained on 450 rows, `Proximity(rf, "gap").fit(X_other)` with 150 rows: returns a 150 × 150 matrix, no error. | Bootstrap statistics are reconstructed for rows the forest never saw. The wrapper should always let ForestGeom fit the forest itself. |
| F3 | The leaf vocabulary is built from the rows passed to `fit`. A query that lands in a leaf none of those rows reached raises `ValueError: Tree 0 returned unknown leaf-node IDs`. | Fitted forest, 20 reference rows, `transform` on the rest: raises. | Only reachable when the reference set is smaller than the forest's training set (fitted forest plus a subsample). The correct proximity in that case is 0. |
| F4 | Dense n × n_trees arrays are held in the cache (`leaf_matrix`, `oob_mask`, `inbag_counts`, three flattened index arrays). | n = 50 000, 500 trees, `gap`: 0.82 GB held, 1.71 GB peak, 13 s to build. This is linear in n × n_trees, so about 16 GB held at 10⁶ rows. | Fine for tens of thousands of rows. A limit for single-cell atlases. |
| F5 | `.github/workflows/test.yml` runs on `workflow_dispatch` only; `push` and `pull_request` are commented out. | Read from the file. | Tests do not run on pull requests upstream. |

Fix sketch for F1: take the in-bag indices from scikit-learn's public
`forest.estimators_samples_`, or pass
`_get_n_samples_bootstrap(n_samples, forest.max_samples)` in place of `n_samples`. The first
also removes two private imports.

### Licence

ForestGeom is GPL-3.0-or-later. manylatents is Apache-2.0. The plan below imports ForestGeom
lazily from an optional extra and copies none of its code. manylatents has been here before:
`leidenalg` is GPL, and PR #336 added a permissive path next to it. Whether an optional GPL
import is acceptable for manylatents is the maintainers' call; whether ForestGeom could be
relicensed is its authors' call (question 1).

---

## 2. Audit of manylatents for supervised work

| # | Gap | Measured |
|---|---|---|
| M1 | In `mode="split"`, labels never reach the module. `run_experiment` reads `train_dataset.get_labels`, and a `random_split` `Subset` has no such method. | `run(data="gaussian_blob", data_kwargs={"n_samples": 400, "mode": "split"})`: the module's `fit` receives `y=None`. In `mode="full"` it receives 400 labels. |
| M2 | `run(input_data=X)` has no way to pass labels. `PrecomputedDataModule` takes labels only from a file. | `data_kwargs={"labels": y}` raises `TypeError`. |
| M3 | With no labels, the in-memory dataset's `get_labels()` returns `np.zeros(n)` (`data/precomputed_dataset.py`), and the engine hands that to the module as real labels. | `ClassifierModule` receives 200 labels with one class. |
| M4 | The metric protocol is `(embeddings, dataset, module, cache)`: one set of rows. The RF-AE scores compare **test rows against train rows** in both spaces and need train labels. `metrics/reference_geometry.py` already says the protocol "has no slot" for a second array. | In split mode the train embedding is never computed: the engine runs `fit(train)` then `transform(test)`. |
| M5 | No k-NN accuracy metric. `Trustworthiness`, `KNNPreservation` and `PearsonCorrelation` score a set against itself. No Kendall-τ importance alignment. | 125 registered names. The label-aware ones (`AUC`, `SilhouetteScore`, and those in `stratification.py`, `outlier_score.py`, `subspace_commitment.py`) all read labels of the evaluated rows only. |
| M6 | `PHATEModule(knn_dist="precomputed_affinity")` fits on a proximity matrix but cannot embed new rows. | `transform` raises `Cannot transform additional data using a precomputed distance matrix`. |
| M7 | No labelled tabular benchmark dataset in core. The core datamodules are synthetic, text, reasoning traces, or precomputed files. | `manylatents/configs/data/`. |
| M8 | Pre-fit sampling applies the train indices to the test tensor and to the output labels (`experiment.py`, step 4c). That is right only when train and test are the same rows. | Read from the code. In `mode="split"` the test rows are indexed with indices drawn for another dataset. |

### Feasibility check

RF-GAP proximities (500 trees, symmetrised, adjusted diagonal) into
`PHATEModule(knn_dist="precomputed_affinity")`, with test rows placed by the linear kernel
extension `Z_test = P_test,train · Z_train`. Data: scikit-learn digits plus 200 noise
features, 80/20 stratified split.

| | Train embedding, 10-NN CV accuracy | Test k-NN accuracy (k from 5 to √n_train) |
|---|---|---|
| RF-GAP → PHATE, linear extension | 0.858 | 0.712 |
| PHATE on features | 0.935 | 0.917 |
| The forest itself | — | 0.956 |

The wiring works in 5 s. The numbers are **not** a result: this is PHATE's defaults on a raw
proximity matrix, not the RF-PHATE recipe, on one dataset and one seed. Two things are worth
reading from it. The recipe matters (question 2). And the linear extension loses 15 points
between train and test, with each test row spreading 90% of its proximity mass over about 282
training rows, which is the weakness RF-AE was built to fix.

---

## 3. Plan

Five phases. Each is one or more PRs that pass CI by themselves. Phases 1 and 4 both depend
on phase 0 and on nothing else, so they can proceed in parallel.

### Design decisions

Four choices shape the phases. Each was weighed against heavier alternatives; the
alternatives are listed so they are not proposed again without a new reason.

| Decision | Chosen | Rejected, and why |
|---|---|---|
| Supervision target | `y` stays a plain array. A `target` argument (an array or a metadata column name) says which array supervises the fit. `get_labels()` keeps its other job, colouring plots. | A typed target carrying "classification" or "regression": the module decides the task, as in scikit-learn. A supervised datamodule base class: it would change every datamodule and the omics extension's contract. |
| Train rows in evaluation | The engine builds one `Split` object (train and test features, targets, embeddings) and passes it to metrics that ask for it. | Storing fit data on the module: metrics get `module=None` for Lightning algorithms. Library functions only: scores would not appear in `run()` results or sweeps. |
| Proximity into an embedder | Nesting: a wrapper module holds a geometry module and an embedder, as `SelectiveCorrectionModule(inner=...)` already does. | One module per pairing (RF-PHATE, RF-UMAP, …): repeats the forest logic. A formal proximity-provider protocol: premature with one provider. A sequential pipeline: RF-AE needs a side input, so a chain is not enough. |
| RF-AE's base class | `LatentModule`, with the network trained inside `fit`. | `LightningModule`: labels would have to come through the datamodule, the engine calls `encode` on raw test features, and the forest would have to be pickled into checkpoints. `ARCHITECTURE.md` says the choice of base is about API surface, not gradients. |

`Split` and `target` are not specific to forests. The pop-gen datamodule in
manylatents-omics loads predefined fit and projection sets as separate train and test
datasets, so its runs can receive a `Split`, and `target="Population"` selects among its
label columns. One change to omics is needed first: `ManifoldGeneticsDataset` builds its
label encoder per dataset from that dataset's own sorted names, so the train set and the
test set can give different integers to the same population. Until the two datasets share
one encoder, a label-based score on an omics `Split` is not valid.

### Constraints found in review

These are facts about the current code that the phases must respect.

- **`target` supervises LatentModules only.** A LightningModule trains from datamodule
  batches, so the engine rejects `target` with one. Existing label-aware metrics keep
  reading the dataset's labels; only metrics that take `split` see `target`.
- **A `Split` needs one fit.** When `transform` raises `NotImplementedError`, the engine
  refits on the test rows. No `Split` is built on that path.
- **Full mode is in-sample.** `Split.held_out` is `False` when the train rows and the test
  rows are the same. Phase 4 metrics refuse such a split and do not report training-set
  accuracy as a score.
- **`fit_transform` must be overridden** by the phase 1 and phase 2 modules. The inherited
  one calls `fit` then `transform`, which for `gap` returns the query extension for the fit
  rows, and full mode uses `fit_transform`.
- **Nesting through the Python API.** `_instantiate_target` in `manylatents/api.py` does not
  instantiate child dicts. Phase 2 either makes it recursive or documents that the API takes
  module instances.
- **Dense outputs are collected on every run.** `collect_outputs` calls `kernel()` and
  `affinity()` whenever they exist. Phase 1 must make the dense matrices opt-in, or the
  "matrix-free" module allocates n × n in ordinary runs.
- **RF-AE.** The geometry must be fitted once and shared with the nested target embedder.
  The inner trainer needs an explicit contract for logger and accelerator, because `fit`
  receives only `x` and `y`. The existing `Autoencoder` maps any activation it does not
  know, including `"elu"`, to ReLU, and has no softmax output.

Revisit the heavier options when a second supervised family arrives, or when repeated or
cross-validated runs are wanted inside `run()`. If a data abstraction is built then, it
should cover per-row annotations in general (labels, time, metadata), not labels alone.

### Phase 0 — supervision plumbing (no ForestGeom)

- Pre-fit sampling slices only the rows it sampled (M8).
- Labels reach the module in split mode (M1).
- `target`: `run(..., target=y)` or `run(..., target="column")` chooses the supervision
  array (M2). Without it, the dataset's own labels are used, as today.
- A dataset with no labels supplies none (M3). The zeros stay available to plotting
  callbacks only.
- **`Split` (M4).** A frozen dataclass with `x_train`, `y_train`, `z_train`, `x_test`,
  `y_test`, `z_test`, and a `held_out` flag. A metric that declares a `split` parameter
  receives it.
  The engine builds it only when a requested metric asks, found by a signature scan of the
  metric functions. `z_train` comes from a new optional module hook,
  `training_embedding(x)`, which defaults to `transform(x)`. The hook exists because
  for bootstrap-dependent schemes the fitted train embedding is not `transform(x_train)`:
  ForestGeom documents that `fit_transform` and `fit().transform` differ for `gap`.
- A `stratify` option (by the dataset's labels or by a named column) on the datamodules
  that split discrete labels, built on a shared helper that extension datamodules can
  reuse. The RF-AE protocol uses stratified 80/20 splits; `random_split` is not stratified.

### Phase 1 — `ForestProximityModule` (LatentModule, extra `forest`)

- New extra: `forest = ["forestgeom>=0.4"]`, imported inside methods.
- Parameters: `forest` (`rf`, `extra_trees`, `random_trees`, `gbt`, `xgb`, `lgbm`, or an
  estimator instance), `task` (`classification` or `regression`, required for supervised
  forests), `weight_scheme`, `n_estimators`, `min_samples_leaf`, `random_state`, `n_jobs`.
- Output: the top `n_components` singular directions of the sparse map `Q` ("leaf PCA", as
  in ForestGeom's `demo_leaf_pca`). It is matrix-free and has a native out-of-sample
  `transform` (`query_map(X_new) · V`).
- `proximity(x=None)`: the sparse train–train block when `x` is `None`, the query–train
  block otherwise. This plain method is what phases 2 and 3 consume.
- `kernel()` and `affinity()` return the dense train–train proximity, so every existing
  `at: kernel` and `at: affinity` metric applies to a supervised geometry.
- Guards from the audit: the wrapper fits the forest itself (F2) and rejects `max_samples`
  with `oob` or `gap` (F1).
- `random_trees` needs no labels, so the same module is also an unsupervised baseline.

### Phase 2 — `ForestGuidedModule` (LatentModule): RF-PHATE and its siblings

- `ForestGuidedModule(geometry=<module with proximity()>, embedder=<LatentModule>)`. Both
  are nested Hydra nodes, as in `configs/algorithms/latent/selective_correction.yaml`.
- `fit(x, y)`: fit the geometry, take the symmetrised train–train proximity, and hand it to
  the embedder. PHATE receives it as an affinity (`knn_dist="precomputed_affinity"`). UMAP,
  t-SNE and Isomap receive a distance (`input_metric="precomputed"`), by default
  `max(P) − P`; the conversion is a parameter.
- `transform(x)`: the linear kernel extension, `Z = rownorm(P(x, train)) · Z_train`. It is
  written once and serves every embedder. `training_embedding(x)` and `fit_transform`
  return the fitted `Z_train`.
- Shipped config: `rf_phate.yaml` (`gap` forest geometry, PHATE embedder).
- Limit: the handoff is a dense n × n matrix.
- If the RF-PHATE recipe cannot be expressed as "PHATE on a precomputed affinity"
  (question 2), `rf_phate` becomes a dedicated module that wraps `rfphate`, and
  `ForestGuidedModule` stays for the other embedders.

### Phase 3 — `RFAEModule` (LatentModule)

- `fit(x, y)`, in stages:
  1. Fit the geometry module (`gap`); take the symmetrised proximity with adjusted diagonal.
  2. Choose N\* prototypes by class-wise k-medoids on `max(P) − P`.
  3. Build each training row: proximities to the prototypes, normalised to sum to 1.
  4. Fit the nested target embedder (the phase 2 module) to get `z_G`.
  5. Train the autoencoder with loss `λ · JSD(p, p̂) + (1 − λ) · ‖z − z_G‖²`.
- `transform(x)`: `proximity(x)` restricted to the prototypes, normalised, then the encoder.
- Defaults from the paper: λ = 0.01, N\* = 0.1 · n_train, encoder 800-400-100-2, ELU,
  softmax output, AdamW (lr 1e-3, weight decay 1e-5), batch 512, 200 epochs.
- Training runs an inner `lightning.Trainer` over a private LightningModule, so loss curves
  still reach the configured logger. The network reuses
  `algorithms/lightning/networks/autoencoder.py` where it fits.
- New pieces: a JSD loss and a k-medoids dependency (the paper uses FasterPAM).

### Phase 4 — RF-AE evaluation metrics (needs phase 0 only)

All are scored on test rows against train rows, read from `split`. K runs from 5 to
√n_train in steps of 10, and scores are averaged over K. Each score is first a plain
function of arrays in `manylatents/metrics/`, then a registered metric that wraps it.

| Metric | Definition |
|---|---|
| `knn_accuracy` | Test accuracy of a k-NN classifier fitted on the train embedding. |
| `oos_qnx`, `oos_trustworthiness` | Local: neighbours of a test row among train rows, feature space against embedding. |
| `oos_spearman`, `oos_pearson` | Global: correlation of the flattened test–train distance matrices. |
| `classification_importance` | C_i = accuracy drop on the test set when feature i is perturbed (correlation-aware: each other feature j is also resampled with probability \|corr(i, j)\|). The classifier is a majority vote of k-NN (k = √n_train), LinearSVC and a two-layer MLP. |
| `structural_importance` | S_i = s(D, D_emb) − s(D̃⁽ⁱ⁾, D_emb): the drop in a structure score s when feature i is perturbed in the test features only. The embedding is held fixed. |
| `sia` | Kendall τ between C and S. Four variants, one per structure score. |

These definitions come from a machine summary of the paper's HTML. I could not read
Algorithm 1 or the appendix formulas directly, so each needs checking against the reference
code before it is written (question 3).

Cost to design for: S needs one perturbed test–train distance matrix per feature. For
squared Euclidean distance this is an update over the changed columns, not a recomputation.

### Phase 5 — labelled datasets

A tabular datamodule with labels, a stratified or predefined split, and min-max scaling,
covering the paper's 20 datasets or a subset. Its source depends on question 5.

---

## 4. Open questions for Adrien

1. **Licence.** Is GPL-3.0 a deliberate choice for ForestGeom, and would a permissive
   licence be possible? It decides whether ForestGeom stays an optional import or can become
   a core dependency.
2. **RF-PHATE recipe.** What are the exact steps from RF-GAP proximities to the RF-PHATE
   embedding (normalisation, diffusion time, any kNN or decay step)? Can it be written as
   PHATE on a precomputed affinity, which is what phase 2 assumes? Should manylatents wrap
   `rfphate`, or build on ForestGeom and its own PHATE? Is there a reference RF-AE
   implementation we should port, or should we write it from the paper?
3. **Metrics.** Are the definitions in phase 4 right? Is the evaluation code public? Which
   classification-importance strategy is the default to ship (ensemble, k-NN alone,
   aggregated)? How did you keep the per-feature structural importance affordable at
   784 features?
4. **Proximity inputs.** Is `training_proximity(adjust_diagonal=True, force_symmetric=True)`
   exactly the p′ of the RF-AE paper, with row normalisation left to the caller? For
   test rows, is `transform(X_test)` (all trees treated as out-of-bag) the proximity RF-AE
   feeds its encoder?
5. **Datasets.** Is there a loader or a data release for the 20 benchmark datasets and their
   splits? `dataset.py` in ForestGeom is GPL, so we would not copy it.
6. **Upstream fixes.** Do you want a patch for F1 (`max_samples`) and guards for F2 and F3?
   I can send them to ForestGeom.
7. **Scope.** Which of `uniform`, `kerf`, `oob`, `boosted` are worth exposing in manylatents
   beyond `gap`, and is there a planned ForestGeom API change (node or path geometry, GPU)
   we should wait for?

---

## 5. Implementation handoff

The implementation will be done by a coding agent (Codex) that starts with none of the
context above. Each task below is written to be handed over as it stands: one task, one
branch, one PR.

**Rules for every task**

- Read `AGENTS.md`, then `CLAUDE.md`, `ARCHITECTURE.md` and `CONTRIBUTING.md`.
- Run with `uv run`. Before pushing:
  `uv run pytest tests/ -x -q && uv run pytest manylatents/callbacks/tests/ -x -q`.
- Keep the change to the files named in the task. Do not add experiment configs, sweeps or
  analysis scripts.
- A measurement that cannot be made raises
  `manylatents.utils.exceptions.MeasurementUnavailable`. It never returns NaN or a
  placeholder.
- Each task lists **new-behaviour tests** and **compatibility tests**. A new-behaviour test
  must fail on the task's base branch and pass on the task branch. A compatibility test
  passes on both.
- With none of the new arguments set, results are identical to the base branch, except for
  the **intended change** that the task names.

**What is ready and what is blocked**

| Task | Base | State |
|---|---|---|
| T0.0 pre-fit sampling in split mode | `main` | Ready. |
| T0.1 labels in split mode | `main` | Ready. |
| T0.2 `target` | T0.1 | Ready. |
| T0.3 no fabricated labels | T0.2 | Ready. |
| T0.4 `Split` | T0.0, T0.3 | Ready. |
| T0.5 stratified splits | T0.3 | Ready. |
| Phase 1 | T0.2 | Ready after T0.2. Scheme list waits on question 7; `gap` and `kerf` are safe to start with. |
| Phase 2: `ForestGuidedModule` | phase 1 | Ready after phase 1. The `rf_phate` config waits on question 2. |
| Phase 4: `knn_accuracy`, `oos_*` | T0.4 | Ready after T0.4, once question 3 confirms the K grid and averaging. |
| Phase 4: importances and `sia` | | Blocked on question 3. |
| Phase 3 | | Blocked on questions 2 and 4. |
| Phase 5 | | Blocked on question 5. |

The six phase 0 tasks are implemented as one stack, in the order T0.0 to T0.5, each branch
on top of the one before it. Briefs for phases 1 to 5 are written when their blocking
questions are answered.

### T0.0 — pre-fit sampling slices only the rows it sampled

- **Problem:** step 4c of `run_experiment` (`manylatents/experiment.py`) draws
  `pre_fit_indices` from the train rows, then applies them to `train_tensor`,
  `test_tensor`, `train_labels` and `output_labels`. When the test dataset is a different
  set of rows, the test tensor and the output labels are indexed with indices that do not
  belong to them.
- **Change:** compute `same_dataset = datamodule.test_dataset is datamodule.train_dataset`
  once. Slice `test_tensor` and `output_labels` by `pre_fit_indices` only when
  `same_dataset` is true. `train_tensor` and `train_labels` are always sliced.
- **Intended change:** with a `"dataset"` sampler and separate test rows, the result has
  one embedding row per test row.
- **Files:** `manylatents/experiment.py`; new `tests/test_prefit_sampling_split.py`.
- **New-behaviour test:** build `GaussianBlobDataModule(n_samples=400, mode="split")` and
  call `run_experiment` with a recorder LatentModule and a sampler object whose
  `get_indices` returns the first 100 indices. `fit` receives 100 rows, and
  `result["embeddings"]` has 80 rows that equal `transform` of the 80 test rows.
- **Compatibility test:** the same sampler in `mode="full"` gives 100 embedding rows, as on
  `main`.

### T0.1 — labels reach the module in split mode

- **Problem:** `run_experiment` (`manylatents/experiment.py`, step 4b) reads
  `train_dataset.get_labels` and `test_dataset.get_labels`. In `mode="split"` both datasets
  are `torch.utils.data.Subset`, which has no `get_labels`, so `algorithm.fit(x, None)` is
  called and the returned `label` is `None`.
- **Change:** for the two `hasattr(..., "get_labels")` reads only, look through
  `_unwrap_for_metrics(dataset)`. `_SubsetView` already realigns `get_labels` by
  `subset.indices`. Do not rebind `train_dataset`: the sampler in step 4c must still
  receive the original object.
- **Intended change:** in split mode `fit` receives the train labels, and
  `result["label"]` holds the test labels where it was `None`.
- **Files:** `manylatents/experiment.py`; new `tests/test_supervised_labels.py`.
- **New-behaviour test:** build `GaussianBlobDataModule(n_samples=400, mode="split")`
  directly and call `run_experiment` with a LatentModule that records `y`, so the test can
  read `datamodule.train_dataset.indices`. Assert `fit` received 320 labels, that they
  equal the full dataset's labels at those indices, and that `result["label"]` equals the
  full labels at `datamodule.test_dataset.indices`.
- **Compatibility test:** in `mode="full"`, `fit` receives 400 labels and `result["label"]`
  has 400 entries.

### T0.2 — `target`: choose the supervision array

- **Problem:** the array passed to `fit` as `y` is whatever `get_labels()` returns.
  `run(input_data=X)` has no way to supply one, and a dataset with several label columns
  has no way to choose.
- **Change:**
  1. Add `target=None` to `run()` (`manylatents/api.py`) and to `run_experiment`, and a
     top-level `target: null` key to the Hydra config (`manylatents/configs/config.py`,
     forwarded in `manylatents/main.py`).
  2. A resolver in `manylatents/utils/data.py`:
     `resolve_target(target, dataset, datamodule=None) -> np.ndarray`. It returns one entry
     per row of `dataset`. T0.5 reuses it. It never returns a tensor, and it returns string
     labels as they are.
     - Let `base` and `indices` be `dataset.dataset` and `dataset.indices` when `dataset`
       is a `torch.utils.data.Subset`, and `dataset` and `None` otherwise. A `Subset` of a
       `Subset` raises `ValueError`.
     - **Array.** Its length must equal `len(base)`: one entry per row of the full
       dataset, before any split. The result is `array[indices]`, or the array when
       `indices` is `None`. A wrong length raises `ValueError` that states both lengths.
     - **String.** The first rule that applies gives a full-length array, which is then
       sliced by `indices`:
       (a) `base.get_labels` takes a parameter (`inspect.signature` on the base's method,
       not on a `_SubsetView`): `base.get_labels(name)`. The pop-gen
       `ManifoldGeneticsDataset.get_labels(label_col)` in manylatents-omics is this case.
       (b) `datamodule` has `get_metadata_column` and its `metadata` has `len(base)` rows:
       `datamodule.get_metadata_column(name)`. `PrecomputedDataModule` is this case; its
       metadata is on the datamodule, not on the dataset.
       (c) `base.metadata` is a `pandas.DataFrame` or a dict that has `name`: that column.
       Otherwise raise `ValueError` that lists the columns found by (b) and (c).
  3. In `run_experiment` step 4b, when `target` is not `None`, the train array is
     `resolve_target(target, train_dataset, datamodule)` and replaces `train_labels`
     before pre-fit sampling, which already slices it. A numeric array becomes a tensor,
     as labels do today; a non-numeric array stays a numpy array. The test array is
     `resolve_target(target, test_dataset, datamodule)`; keep it in a local `test_target`
     for T0.4.
  4. An array `target` needs the train and test datasets to share one base: the same
     object, or two `Subset`s of the same dataset. Otherwise raise `ValueError` that says
     to pass a column name.
  5. `target` with a LightningModule raises `ValueError`: a LightningModule trains from
     datamodule batches, so `target` would not supervise it.
  6. In the transductive fallback (`transform` raises `NotImplementedError`), `fallback_y`
     is `test_target` when `target` is given.
  7. `target` changes only what `fit` and, after T0.4, `Split` receive. `result["label"]`,
     plotting and the existing label-aware metrics are unchanged.
- **Files:** `manylatents/api.py`, `manylatents/experiment.py`, `manylatents/main.py`,
  `manylatents/configs/config.py`, `manylatents/utils/data.py`,
  `tests/test_supervised_labels.py`, `docs/api_usage.md`.
- **New-behaviour tests:**
  - `run(input_data=X, target=y, algorithm=<recorder>)` in full mode and with
    `data_kwargs={"mode": "split"}`: `fit` receives `y` row-aligned with `x`. Check by
    making `y` a function of a column of `X`.
  - A wrong length raises. `target` with a LightningModule raises.
  - On `gaussian_blob`, an explicit array wins over the dataset's labels, and
    `result["label"]` is still the dataset's labels.
  - A `PrecomputedDataModule` with a metadata file and `target="<column>"`, full and
    split: `fit` receives that column, row-aligned, including a column of strings.
  - `resolve_target` on a stub dataset whose `get_labels(label_col)` takes a column name
    uses rule (a). An unknown name raises and the message lists the available columns.

### T0.3 — a dataset with no labels supplies none

- **Problem:** the three dataset classes in `manylatents/data/precomputed_dataset.py` return
  `np.zeros(len(self.data))` from `get_labels()` when they have no labels. The engine passes
  that to `fit` as real labels. Measured: `ClassifierModule` receives 200 labels with one
  class.
- **Change:** add a `has_labels` property to those three classes. It is `False` exactly
  when `get_labels()` would fabricate zeros. It depends on whether labels were supplied,
  not on their values: real labels that are all zero give `True`. Let `_SubsetView`
  forward `has_labels` unsliced. In `run_experiment` step 4b, when `target` is `None` and
  the dataset (seen through `_unwrap_for_metrics`) has `has_labels == False`, set
  `train_labels = None`, and likewise `test_target = None`. A dataset without the
  attribute behaves as today. Leave `get_labels()` and `output_labels` unchanged, because
  plotting callbacks rely on the zeros.
- **Intended change:** `fit` receives `y=None` where it received fabricated zeros.
- **Files:** `manylatents/data/precomputed_dataset.py`, `manylatents/experiment.py`,
  `tests/test_supervised_labels.py`.
- **New-behaviour tests:** `run(input_data=X, algorithm=<recorder>)` gives `fit` a `y` of
  `None`, in full and in split mode. `ClassifierModule` then raises its own "requires
  labels" error. `has_labels` is `False` for each of the three classes built without
  labels, and `True` for a `PrecomputedDataset` whose label column is all zeros.
- **Compatibility test:** `result["label"]` for `run(input_data=X, ...)` is the zeros
  array, as on the base branch.

### T0.4 — `Split`: train rows for metrics

- **Problem:** a metric receives `(embeddings, dataset, module, cache)`, which describe the
  evaluated rows only. Out-of-sample scores need the train rows in both spaces and both
  target arrays. In split mode the train embedding is never computed.
- **Change:**
  1. `LatentModule.training_embedding(x)` in `latent_module_base.py`: a non-abstract method
     that returns `self.transform(x)`. Modules whose fitted embedding differs from
     `transform` of the fit rows override it.
  2. `Split` in `manylatents/evaluate.py`: a frozen dataclass with `x_train`, `y_train`,
     `z_train`, `x_test`, `y_test`, `z_test` and `held_out: bool`. The six arrays are numpy
     arrays (tensors are converted with `.detach().cpu().numpy()`); the two `y` fields may
     be `None`. Rows of the three train fields correspond, and so do rows of the three test
     fields. `held_out` is `False` when the train rows are the test rows. Metrics must not
     write to the arrays.
  3. `metrics_declaring(metrics, name) -> list[str]` in `manylatents/evaluate.py`: the
     requested metrics whose function has a parameter called `name`. On the registry path
     inspect `get_metric(metric).func`. On the Hydra path resolve `_target_` with
     `hydra.utils.get_object` and inspect that. `extract_k_requirements` does not do this
     for Hydra configs, so this helper is new.
  4. `evaluate()` takes `split: Split | None = None` and passes it to exactly the metrics
     that `metrics_declaring(metrics, "split")` names, on both paths. Other metrics are
     called exactly as today. A metric that declares `split` when `split` is `None` is not
     called: its result is `MeasurementUnavailable`, handled as `evaluate()` already
     handles that exception from a metric.
  5. `run_experiment` builds the `Split` only when `metrics_declaring(metrics, "split")`
     is not empty.
     - `y_train` is the `train_labels` that `fit` received. `y_test` is `test_target` when
       `target` is given; otherwise the test dataset's labels, or `None` when the dataset
       has none (T0.3).
     - **Separate test rows** (`datamodule.test_dataset is not datamodule.train_dataset`):
       `held_out=True`. `x_train` is the fit tensor after pre-fit sampling. `z_train` is
       `algorithm.training_embedding(x_train)` for a LatentModule and
       `algorithm.encode(x_train)` for a LightningModule, called the way the engine
       already calls `encode(test_tensor)`.
     - `x_train` must be in **dataset order**. For a LightningModule the engine's existing
       `train_tensor` comes from the shuffled train loader and must not be used here;
       unroll the train dataset with `shuffle=False`, as the LatentModule branch does.
     - **Same rows** (full mode): `held_out=False`, the train fields are the same objects
       as the test fields, and the already computed embedding is reused.
       `training_embedding` is not called.
     - **Transductive fallback** (`transform` raised `NotImplementedError` and the engine
       refitted on the test rows): no `Split` is built, and `split=None` goes to
       `evaluate()`.
     - **Post-fit sampling** (a `sampling` key other than `"dataset"`) together with a
       metric that declares `split` raises `ValueError`. The two are not combined in this
       task.
- **Files:** `manylatents/algorithms/latent/latent_module_base.py`,
  `manylatents/evaluate.py`, `manylatents/experiment.py`, `manylatents/metrics/metric.py`
  (document the parameter), `docs/evaluation.md`, new `tests/test_metric_split.py`.
- **New-behaviour tests:**
  - Register a test-only metric that returns facts about `split`. In split mode on
    `gaussian_blob` with 400 samples: `z_train` has 320 rows, `y_train` equals the labels
    `fit` received, `y_test` has 80 entries, `held_out` is `True`.
  - The same metric given as a Hydra-style config dict with `_target_` also receives the
    split.
  - Full mode: `held_out` is `False` and `z_train is z_test`.
  - A module that overrides `training_embedding` to return a marker array: `z_train` is
    that array.
  - With a `"dataset"` sampler in split mode, the train fields have the sampled row count
    and the test fields have 80 rows.
  - A module whose `transform` raises `NotImplementedError`, in split mode: the metric's
    result is `MeasurementUnavailable`.
  - With a LightningModule: instantiate the `ae_reconstruction` algorithm and call
    `run_experiment` with `Trainer(fast_dev_run=True)` (`run()` does not expose trainer
    settings). `split.x_train` equals the train dataset's rows in dataset order.
- **Compatibility tests:** a run whose metrics do not declare `split` never calls
  `training_embedding` (a module that raises from it). Existing metric tests pass
  unchanged.

### T0.5 — stratified splits

- **Problem:** split-mode datamodules call `torch.utils.data.random_split`, which ignores
  class balance. The RF-AE protocol uses a stratified 80/20 split.
- **Change:**
  1. A shared helper in `manylatents/utils/data.py`:
     `stratified_subsets(dataset, labels, test_split, seed) -> (train_subset, test_subset)`.
     It builds two `torch.utils.data.Subset`s from
     `sklearn.model_selection.train_test_split(np.arange(n), test_size=test_split,
     stratify=labels, random_state=seed)`. Split sizes follow scikit-learn's rounding. A
     class with one member raises scikit-learn's `ValueError`, which is not caught. The
     helper is public so that extension datamodules (manylatents-omics) can call it; this
     task does not change any of them.
  2. Add `stratify: bool | str = False` to `GaussianBlobDataModule` and
     `PrecomputedDataModule`. In `mode="split"`:
     - `True` stratifies by `full_dataset.get_labels()`. It raises `ValueError` when the
       dataset has `has_labels == False` (T0.3).
     - A string stratifies by `resolve_target(name, full_dataset, datamodule=self)`
       (T0.2). This is allowed when the dataset has no labels of its own.
     - `False` leaves the code path and the resulting split unchanged.
     - The seed is the one each datamodule already gives to `random_split`:
       `self.random_state` for Gaussian blobs, `self.hparams.seed` for precomputed data.
  3. `stratify` other than `False` in `mode="full"` raises `ValueError`.
  4. Add `stratify: false` to the two data configs.
- **Files:** `manylatents/utils/data.py`, `manylatents/data/gaussian_blobs.py`,
  `manylatents/data/precomputed_datamodule.py`, `manylatents/configs/data/gaussian_blobs.yaml`,
  `manylatents/configs/data/precomputed.yaml`, new `tests/test_stratified_split.py`.
- **New-behaviour tests:** on imbalanced blobs, class proportions in train and test match
  the full dataset to within one sample per class with `stratify=True`. With a
  `PrecomputedDataModule` that has a metadata file and no label column,
  `stratify="<column>"` balances that column. `stratify=True` on in-memory data with no
  labels raises, and so does `stratify=True` in `mode="full"`. The helper, called directly
  on a plain dataset and a label array, returns disjoint subsets that cover every row, and
  raises for a class with one member.
- **Compatibility test:** with `stratify=False`, the train and test indices equal those on
  the base branch for the same seed (compare with a direct `random_split` call that uses
  the same generator seed).
