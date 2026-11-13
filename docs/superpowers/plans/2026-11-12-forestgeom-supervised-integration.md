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
"embedding" is one column, P(y=1|x), scored by one metric, `AUC`. It handles binary labels
only.

Everything in the "measured" tables was run on 2026-11-12 against ForestGeom 0.4.0 (commit
`053ef74`) and manylatents `ad0e10e`, with scikit-learn 1.8.0.

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
| M1 | In `mode="split"`, labels never reach the module. `run_experiment` reads `train_dataset.get_labels`, and a `random_split` `Subset` has no such method. | `run(data="gaussian_blob", data_kwargs={"mode": "split"})`: the module's `fit` receives `y=None`. In `mode="full"` it receives 400 labels. |
| M2 | `run(input_data=X)` has no way to pass labels. `PrecomputedDataModule` takes labels only from a file. | `data_kwargs={"labels": y}` raises `TypeError`. |
| M3 | With no labels, the in-memory dataset's `get_labels()` returns `np.zeros(n)` (`data/precomputed_dataset.py`), and the engine hands that to the module as real labels. | `ClassifierModule` receives 200 labels with one class. |
| M4 | The metric protocol is `(embeddings, dataset, module, cache)`: one set of rows. The RF-AE scores compare **test rows against train rows** in both spaces and need train labels. `metrics/reference_geometry.py` already says the protocol "has no slot" for a second array. | In split mode the train embedding is never computed: the engine runs `fit(train)` then `transform(test)`. |
| M5 | No k-NN accuracy metric. `Trustworthiness`, `KNNPreservation` and `PearsonCorrelation` score a set against itself. No Kendall-τ importance alignment. | 125 registered names; the label-aware ones are `AUC` and `SilhouetteScore`. |
| M6 | `PHATEModule(knn_dist="precomputed_affinity")` fits on a proximity matrix but cannot embed new rows. | `transform` raises `Cannot transform additional data using a precomputed distance matrix`. |
| M7 | No labelled real dataset in core. All core datamodules are synthetic. | `manylatents/configs/data/`. |

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

Revisit the heavier options when a second supervised family arrives, or when repeated or
cross-validated runs are wanted inside `run()`. If a data abstraction is built then, it
should cover per-row annotations in general (labels, time, metadata), not labels alone.

### Phase 0 — supervision plumbing (no ForestGeom)

- Labels reach the module in split mode (M1).
- `target`: `run(..., target=y)` or `run(..., target="column")` chooses the supervision
  array (M2). Without it, the dataset's own labels are used, as today.
- A dataset with no labels supplies none (M3). The zeros stay available to plotting
  callbacks only.
- **`Split` (M4).** A frozen dataclass with `x_train`, `y_train`, `z_train`, `x_test`,
  `y_test`, `z_test`. A metric that declares a `split` parameter receives it. The engine
  builds it only when a requested metric asks, by the same signature scan that
  `extract_k_requirements` uses for `k`. `z_train` comes from a new optional module hook,
  `training_embedding()`, which defaults to `transform(x_train)`. The hook exists because
  for bootstrap-dependent schemes the fitted train embedding is not `transform(x_train)`:
  ForestGeom documents that `fit_transform` and `fit().transform` differ for `gap`.
- A `stratify` option on the datamodules that split discrete labels. The RF-AE protocol
  uses stratified 80/20 splits; `random_split` is not stratified.

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
  written once and serves every embedder. `training_embedding()` returns the fitted
  `Z_train`.
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
- Every new test must fail on `main` and pass on the branch.
- With none of the new arguments set, results must be identical to `main`.

**What is ready and what is blocked**

| Task | State |
|---|---|
| T0.1 – T0.5 (phase 0) | Ready now. No dependency on ForestGeom or on Adrien's answers. T0.2 and T0.3 build on T0.1; T0.4 builds on T0.2. |
| Phase 1 | Ready after T0.1 and T0.2. Scheme list waits on question 7; `gap` and `kerf` are safe to start with. |
| Phase 2: `ForestGuidedModule` | Ready after phase 1. The `rf_phate` config waits on question 2. |
| Phase 4: `knn_accuracy`, `oos_*` | Ready after T0.4, once question 3 confirms the K grid and averaging. |
| Phase 4: importances and `sia` | Blocked on question 3. |
| Phase 3 | Blocked on questions 2 and 4. |
| Phase 5 | Blocked on question 5. |

Briefs for phases 1 to 5 are written when their blocking questions are answered.

### T0.1 — labels reach the module in split mode

- **Problem:** `run_experiment` (`manylatents/experiment.py`, step 4b) reads
  `train_dataset.get_labels` and `test_dataset.get_labels`. In `mode="split"` both datasets
  are `torch.utils.data.Subset`, which has no `get_labels`, so `algorithm.fit(x, None)` is
  called and the returned `label` is `None`.
- **Change:** pass both datasets through the existing `_unwrap_for_metrics` before the
  `hasattr(..., "get_labels")` checks. `_SubsetView` already realigns `get_labels` by
  `subset.indices`.
- **Files:** `manylatents/experiment.py`; new `tests/test_supervised_labels.py`.
- **Test:** `run(data="gaussian_blob", algorithm=<LatentModule that records y>,
  data_kwargs={"n_samples": 400, "mode": "split"})`. Assert `fit` received 320 labels, that
  they equal the full dataset's labels at the train subset's indices, and that
  `result["label"]` has 80 entries.

### T0.2 — `target`: choose the supervision array

- **Problem:** the array passed to `fit` as `y` is whatever `get_labels()` returns.
  `run(input_data=X)` has no way to supply one, and a dataset with several label columns
  has no way to choose.
- **Change:** add `target=None` to `run()` (`manylatents/api.py`) and to `run_experiment`,
  and a top-level `target: null` key to the Hydra config (`manylatents/configs/config.py`,
  forwarded in `manylatents/main.py`).
  - An array must have one entry per row of the full dataset, before any split. The engine
    aligns it to the train and test rows: by `subset.indices` when the dataset is a
    `Subset`, unchanged otherwise. A wrong length raises `ValueError`.
  - A string names a column of the dataset's `metadata`. It is read through the unwrapped
    dataset, so it is already row-aligned. A missing column, or a dataset with no tabular
    `metadata`, raises `ValueError` naming what is available.
  - The resolved train array replaces `train_labels` in step 4b, before pre-fit sampling,
    which already slices it. The resolved test array is kept for T0.4.
  - `target` changes only what `fit` and `Split` receive. `result["label"]` and plotting
    are unchanged.
- **Files:** `manylatents/api.py`, `manylatents/experiment.py`, `manylatents/main.py`,
  `manylatents/configs/config.py`, `tests/test_supervised_labels.py`, `docs/api_usage.md`.
- **Test:** `run(input_data=X, target=y, algorithm=<recorder>)` with the default datamodule
  and with a split one; `fit` receives `y` row-aligned with `x` (check by making `y` a
  function of a column of `X`). Wrong length raises. On `gaussian_blob`, an explicit
  `target` wins over the dataset's labels.

### T0.3 — a dataset with no labels supplies none

- **Problem:** the three dataset classes in `manylatents/data/precomputed_dataset.py` return
  `np.zeros(len(self.data))` from `get_labels()` when they have no labels. The engine passes
  that to `fit` as real labels. Measured: `ClassifierModule` receives 200 labels with one
  class.
- **Change:** add a `has_labels` property to those three classes, `False` exactly when
  `get_labels()` would fabricate zeros. In `run_experiment` step 4b, when no `target` is
  given and the unwrapped dataset has `has_labels == False`, set `train_labels = None`. A
  dataset without the attribute behaves as today. Leave `get_labels()` and `output_labels`
  unchanged, because plotting callbacks rely on the zeros. Let `_SubsetView` forward
  `has_labels` unsliced.
- **Files:** `manylatents/data/precomputed_dataset.py`, `manylatents/experiment.py`,
  `tests/test_supervised_labels.py`.
- **Test:** `run(input_data=X, algorithm=<recorder>)` gives `fit` a `y` of `None`;
  `ClassifierModule` then raises its own "requires labels" error. `result["label"]` is
  unchanged from `main`.

### T0.4 — `Split`: train rows for metrics

- **Problem:** a metric receives `(embeddings, dataset, module, cache)`, which describe the
  evaluated rows only. Out-of-sample scores need the train rows in both spaces and both
  target arrays. In split mode the train embedding is never computed.
- **Change:**
  1. `LatentModule.training_embedding(x)` in `latent_module_base.py`: a non-abstract method
     that returns `self.transform(x)`. Modules whose fitted embedding differs from
     `transform` of the fit rows override it.
  2. `Split` in `manylatents/evaluate.py`: a frozen dataclass with `x_train`, `y_train`,
     `z_train`, `x_test`, `y_test`, `z_test`. All are numpy arrays; the two `y` fields may
     be `None`. Rows of the three train fields correspond, and so do rows of the three test
     fields.
  3. `evaluate()` takes `split: Split | None = None` and passes it to every metric whose
     signature has a parameter named `split`, on both the registry path and the Hydra
     path. Metrics without that parameter are called exactly as today.
  4. `run_experiment` builds the `Split` only when at least one requested metric declares
     `split`. Use `inspect.signature`, as `extract_k_requirements` does for `k`.
     - `z_train` is `training_embedding(x_train)` for a LatentModule and `encode(x_train)`
       for a LightningModule.
     - `x_train` must be unrolled in **dataset order**. For a LightningModule the engine's
       existing `train_tensor` comes from the shuffled train loader and must not be used
       here; unroll the train dataset with `shuffle=False`, as the LatentModule branch
       already does.
     - In `mode="full"` the train fields are the same objects as the test fields and the
       already computed embedding is reused.
     - When pre-fit sampling is active, the train fields are the sampled rows.
  5. A metric that declares `split` and receives `None` raises `MeasurementUnavailable`.
- **Files:** `manylatents/algorithms/latent/latent_module_base.py`,
  `manylatents/evaluate.py`, `manylatents/experiment.py`, `manylatents/metrics/metric.py`
  (document the parameter), `docs/evaluation.md`, new `tests/test_metric_split.py`.
- **Test:** register a test-only metric that returns `split.z_train.shape[0]`. In split
  mode on `gaussian_blob` with 400 samples it returns 320; `split.y_train` equals the labels
  `fit` received; `split.y_test` has 80 entries. A run whose metrics do not declare `split`
  never calls `training_embedding` (assert with a module that raises from it). With a
  LightningModule (`ae_reconstruction`, `fast_dev_run`), `split.x_train` equals the train
  dataset's rows in dataset order. Existing metric tests pass unchanged.

### T0.5 — stratified splits

- **Problem:** split-mode datamodules call `torch.utils.data.random_split`, which ignores
  class balance. The RF-AE protocol uses a stratified 80/20 split.
- **Change:** add `stratify: bool = False` to `GaussianBlobDataModule` and
  `PrecomputedDataModule`. When `True` in `mode="split"`, build the two `Subset`s from
  `sklearn.model_selection.train_test_split(np.arange(n), test_size=test_split,
  stratify=labels, random_state=seed)`. When `False`, the code path and the resulting
  split are unchanged. `stratify=True` on a dataset with no real labels (T0.3) raises
  `ValueError`. Add `stratify: false` to the two data configs.
- **Files:** `manylatents/data/gaussian_blobs.py`,
  `manylatents/data/precomputed_datamodule.py`, `manylatents/configs/data/gaussian_blobs.yaml`,
  `manylatents/configs/data/precomputed.yaml`, new `tests/test_stratified_split.py`.
- **Test:** on imbalanced blobs, class proportions in train and test match the full
  dataset to within one sample per class with `stratify=True`. With `stratify=False`, the
  train and test indices equal those on `main` for the same seed.
