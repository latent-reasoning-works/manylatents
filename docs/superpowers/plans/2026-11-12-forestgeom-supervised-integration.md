# ForestGeom and supervised representation learning in manylatents: audit and plan

**Status:** plan for discussion. No code in this PR. Nothing below is decided until the open
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

### Phase 0 — labels and a train-side reference (no ForestGeom)

- Labels reach the module in split mode (M1), and `run(input_data=X, labels=y)` works (M2).
- A dataset with no labels reports none (M3). The zeros stay available to plotting callbacks
  only.
- **Reference slot (M4).** A metric that declares a `reference` parameter receives
  `{"data": X_train, "labels": y_train, "embedding": Z_train}`. The engine builds it only
  when a requested metric asks for it, by the same signature scan that `prewarm_cache` uses
  for `k`. `Z_train` comes from a new optional module hook `training_embedding()`, which
  defaults to `transform(X_train)`. The hook exists because for bootstrap-dependent schemes
  the fitted train embedding is not `transform(X_train)`: ForestGeom documents that
  `fit_transform` and `fit().transform` differ for `gap`.

### Phase 1 — `ForestProximityModule` (LatentModule, extra `forest`)

- New extra: `forest = ["forestgeom>=0.4"]`, imported inside methods.
- Parameters: `forest` (`rf`, `extra_trees`, `random_trees`, `gbt`, `xgb`, `lgbm`, or an
  estimator instance), `task` (`classification`, `regression`, `auto` from the label dtype),
  `weight_scheme`, `n_estimators`, `min_samples_leaf`, `random_state`, `n_jobs`.
- Output: the top `n_components` singular directions of the sparse map `Q` ("leaf PCA", as
  in ForestGeom's `demo_leaf_pca`). It is matrix-free and has a native out-of-sample
  `transform` (`query_map(X_new) · V`).
- `kernel()` and `affinity()` return the proximity, so every existing `at: kernel` and
  `at: affinity` metric (spectral entropy, spectral gap, kernel sparsity, …) applies to a
  supervised geometry with no further work.
- Guards from the audit: the wrapper fits the forest itself (F2) and rejects `max_samples`
  with `oob` or `gap` (F1).
- `random_trees` needs no labels, so the same module is also an unsupervised baseline.

### Phase 2 — RF-PHATE (LatentModule)

- Proximities from phase 1 into PHATE's diffusion and MDS.
- Out-of-sample: the linear kernel extension, which RF-AE uses as RF-PHATE's baseline.
- Whether this wraps the `rfphate` package or is built on `PHATEModule` depends on
  question 2.

### Phase 3 — RF-AE (LightningModule)

- Input: rows of row-normalised RF-GAP proximities, restricted to N\* prototypes chosen by
  class-wise k-medoids on `max(P) − P`.
- Loss: `λ · JSD(p, p̂) + (1 − λ) · ‖z − z_G‖²`, where `z_G` is the phase 2 embedding.
  Defaults from the paper: λ = 0.01, N\* = 0.1 · n_train, encoder 800-400-100-2, ELU, softmax
  output, AdamW (lr 1e-3, weight decay 1e-5), batch 512, 200 epochs.
- New pieces: a JSD loss, a proximity-row datamodule wrapper, and a k-medoids dependency
  (the paper uses FasterPAM).
- `Reconstruction` cannot be reused unchanged: it reconstructs `batch["data"]` and has no
  per-row target.

### Phase 4 — RF-AE evaluation metrics (needs phase 0 only)

All are scored on test rows against train rows. K runs from 5 to √n_train in steps of 10,
and scores are averaged over K.

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
   embedding (normalisation, diffusion time, any kNN or decay step)? Should manylatents wrap
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
