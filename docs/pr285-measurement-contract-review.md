# PR #285: measurement-contract work stopped

The estimator cannot satisfy its advertised label-specific excess interpretation
through interface and validation fixes. Implementation stopped under the explicit
instruction to stop if making the branch honest requires redesigning its statistic.
This branch is **unfinished and not ready to merge**. No metric behavior was changed.

## Verified blocker

The fitted null and the random-orientation null describe different hypotheses.
For identical label-conditioned distributions confined to the same one-dimensional
axis, every fitted rank-one basis spans that axis. For every nonzero sample, the
projection energies are identical across groups, so normalized weights are 1/K,
entropy is log(K), and observed commitment is zero. Independent random bases
almost surely have unequal squared projections onto that axis, producing positive
commitment. Subtraction therefore gives negative excess despite identical group
distributions. Cross-fitting does not remove this discrepancy.

A direct call to the branch's public SubspaceCommitment function gave:

| Evidence | Mean | Null mean | Excess |
| --- | ---: | ---: | ---: |
| Identical rank-one label distributions | 0 | 0.450069 | -0.450069 |
| Isotropic Gaussian, independent labels | 0.109715 | 0.115173 | -0.005457 |
| Same Gaussian data, first feature scaled by 100 | 0.001229 | 0.121414 | -0.120186 |

These are observed diagnostic outputs, not numerical test tolerances. The
rank-one counterexample follows from the expression above without a tolerance.
Reproduce using the existing interpreter from the repository root:

```python
from types import SimpleNamespace
import numpy as np
from manylatents.metrics import SubspaceCommitment

x = np.zeros((16, 8))
x[:, 0] = np.tile([-4., -3., -2., -1., 1., 2., 3., 4.], 2)
y = np.repeat([0, 1], 8)
print(SubspaceCommitment(x, dataset=SimpleNamespace(metadata=y),
                         rank=1, n_null=100))

x = np.random.default_rng(2).standard_normal((160, 64))
y = np.repeat(np.arange(4), 40)
for data in (x, x * np.r_[100., np.ones(63)]):
    print(SubspaceCommitment(data, dataset=SimpleNamespace(metadata=y),
                             n_null=100))
```

## Rework decision and consumer correction

Defer the estimator redesign and this PR's merge. A label-permutation baseline
that repeats fitting and scoring is a candidate for label-specific excess, but
changes the null hypothesis and requires an explicit exchangeability contract.
Origin handling and the measured population must also be chosen explicitly.
Rejecting nonfinite or rank-deficient inputs cannot correct a baseline mismatch
on finite inputs with fully supported requested rank.

The earlier review's lack of a demonstrated consumer is incomplete. Local
reasoning-geometry contains a related implementation in
experiments/analysis/74_class_commitment_dynamics.py, consumption of its output
in 73_cl_forecast.py, and notes/nizar-task-commitment-block.md explicitly asks to
merge #285. Those notes interpret negative excess under shared collapse as a
substantive reading. There was no direct use of this PR's function found in the
searched local repositories, but the underlying statistic has downstream use.
Replacing the null would change that interpretation. Rework is worth a separate
consumer-informed design decision; it is not justified as an interface repair.

## Named changes still outstanding

- Adopt the exact supplied manylatents/utils/exceptions.py contract and raise
  MeasurementUnavailable for unavailable measurements. That file is absent at
  both this PR head and the available origin/main; no parallel exception was added.
- Validate aligned finite inputs and requested integer parameters; refuse
  unsupported rank, zero total energy, and failed decompositions without silently
  repairing evidence or discarding observations or null draws.
- Define full population/coverage semantics, including small groups. Centered
  m-row data has rank at most min(m - 1, d), because centered rows sum to zero;
  actual numerical rank may be smaller. Default rank four currently produces
  two- and three-column bases with five samples per group, although their
  centered halves support at most one and two dimensions respectively.
- Choose and document origin and null semantics before claiming label-specific
  excess. Concentration does not establish own-label agreement or temporal change.
- Adopt LabeledArray inputs and Table results with an explicit registry adapter,
  validate dimensions and coordinates, and test the dependency compatibility range.
- Declare the intended scalar_key and add the metric discovery YAML once the
  measurement itself is defined. Available origin/main rejects ambiguous dict
  metrics; this branch's decorator has no scalar_key.
- Replace NaN-expecting tests and add regressions for the corrected behavior,
  including anisotropic independent labels, unequal groups, deficient rank,
  nonfinite evidence, scale/origin behavior, and planted structure. No tests were
  rewritten to endorse the existing failures or a redesigned statistic.

Direct probes also confirmed finite output on constant evidence (16 by 8,
rank two), finite output with rank=-1, finite aggregates with an all-NaN sample,
and NaN null/excess for n_null=0. Source inspection confirms nan_to_num, nanmean,
unreported exclusion of small groups, and fitting centered but scoring raw data.

## Verification and Git state

Original head: 35dbc49. Available origin/main: 3be76939dfb7090e519270d9458fa6b0fc324f22.
No network fetch was attempted, as instructed. Interpreter: Python 3.12.13;
NumPy 2.2.6.

Exact requested suite command:

```sh
/Users/cmvcordova/code/lrw/manylatents/.venv/bin/python -m pytest tests/ -q --ignore=tests/test_hf_text_datamodule.py
```

Before: **677 passed, 10 failed, 23 skipped** (126.46 seconds). The six existing
SubspaceCommitment tests passed. Failures were in GPU metric integration,
latent backend/ndarray handling, selective_correction and UMAP instantiation,
t-SNE backend, and UMAP backend tests. The captured UMAP traceback reports
Numba unable to cache in the read-only installed environment.

After: **677 passed, 10 failed, 23 skipped** (70.78 seconds), with the same
failed test names. All ten after-run failures report Numba cache-location errors
in the read-only installed UMAP/pynndescent environment. Full after-run output
is saved locally at /private/tmp/pr285-after-pytest.log. No production or test
code changed, and no successful homogenization is claimed.

Rebase attempt: git rebase origin/main could not create
/Users/cmvcordova/code/lrw/manylatents/.git/worktrees/pr285/rebase-merge
(Operation not permitted). Git did not reach conflict checking and HEAD did not
move. The worktree's shared Git metadata is outside the writable sandbox roots.
Commit blocked: git add docs/pr285-measurement-contract-review.md failed to
create /Users/cmvcordova/code/lrw/manylatents/.git/worktrees/pr285/index.lock
(Operation not permitted). The report remains untracked; no commit was made.
No permission escalation is available.
