# Robust PCA Design Spec

**Date:** 2026-03-25
**Status:** Reviewed

## Overview

Extend the existing `PCAModule` with two robust PCA capabilities:

1. **Global Robust PCA** — Decompose a full matrix D = L + S (low-rank + sparse) via Principal Component Pursuit (Candes et al. 2011). Solves:
   ```
   minimize ||L||_* + lambda ||S||_1   subject to  D = L + S
   ```
   Also supports the **stable variant** (Zhou et al. 2010): `||D - L - S||_F <= delta`.

2. **Robust Local PCA** — Per-neighborhood robust tangent space estimation + Local Tangent Space Alignment (LTSA) for embedding. Protects local geometry estimates against hub points, shortcut neighbors, noise, and embedding distortion.

Both integrate into `PCAModule` via the `method` parameter and are callable from the manylatents API.

Additionally, refactor `compute_knn()` out of `utils/metrics.py` into `utils/knn.py` since it's now used by both metrics and algorithms.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Integration point | Bake into existing `PCAModule` | One PCA class for all PCA-family operations |
| Method selection | `method`: `'standard'`, `'robust_admm'`, `'robust_ialm'`, `'robust_local'` | Extensible, discoverable |
| Solver location | `manylatents/utils/robust_pca_solvers.py` | Keeps `algorithms/latent/` for module classes; solvers are reusable utilities |
| kNN refactor | `manylatents/utils/knn.py` | `compute_knn()` used by metrics + algorithms; `utils/metrics.py` name is misleading |
| Global RPCA transform() | SVD projection of L into n_components dims | "PCA on denoised data" |
| Local RPCA transform() | LTSA alignment of robust local tangent spaces | Principled manifold embedding |
| Extra outputs | L, S, rank, convergence_history; local eigenspectra, dims, outlier masks | Full decomposition accessible via `extra_outputs()` |
| Test strategy | Synthetic ground truth + API integration | No external baseline deps in CI |

## File Structure

```
manylatents/
  utils/
    knn.py                        # NEW: compute_knn() + related (moved from utils/metrics.py)
    robust_pca_solvers.py         # NEW: global RPCA solvers (ADMM, IALM) + robust_local_pca()
    metrics.py                    # MODIFIED: re-export compute_knn from knn.py for backward compat
  algorithms/latent/
    pca.py                        # MODIFIED: method param, robust dispatch
  configs/algorithms/latent/
    pca.yaml                      # Unchanged (method defaults to 'standard')
    robust_pca.yaml               # NEW: method='robust_ialm' + global RPCA params
    robust_local_pca.yaml         # NEW: method='robust_local' + local PCA params
tests/
  test_robust_pca.py              # NEW: global RPCA tests + API integration
  test_robust_local_pca.py        # NEW: local RPCA tests + contaminated manifold recovery
```

No new public exports in `algorithms/latent/__init__.py`.

**Note:** The existing `test_module_instantiation.py` auto-discovers all YAML configs in `configs/algorithms/latent/`. New configs will be auto-discovered, so PCAModule constructor changes must be committed before the config files are added.

---

# Part 1: kNN Refactor

## `utils/knn.py` (new)

Move `compute_knn()` and its helper `_content_key()` from `utils/metrics.py` to `utils/knn.py`. The function signature and behavior are unchanged.

In `utils/metrics.py`, add a backward-compatible re-export:
```python
from manylatents.utils.knn import compute_knn, _content_key  # backward compat
```

This is a pure refactor — no behavior changes, no import breakage.

---

# Part 2: Global Robust PCA

## API: PCAModule Changes

### Constructor

Current signature:
```python
def __init__(self, n_components=2, random_state=42, fit_fraction=1.0, **kwargs)
```

New signature (backward-compatible):
```python
def __init__(self, n_components=2, random_state=42, fit_fraction=1.0,
             method='standard',        # 'standard' | 'robust_admm' | 'robust_ialm' | 'robust_local'
             # --- Global RPCA params (robust_admm / robust_ialm) ---
             lmbda=None,               # lambda for RPCA (default: 1/sqrt(max(m,n)))
             solver_max_iter=500,      # max iterations for robust solver
             tol=1e-7,                 # convergence tolerance
             delta=None,              # stable PCP noise bound (None = exact PCP)
             mu=None,                 # initial penalty param (auto-set if None)
             mu_max=1e7,              # max penalty (IALM)
             rho=1.5,                # penalty growth rate (IALM)
             use_truncated_svd=True,   # adaptive truncated SVD for performance
             # --- Local RPCA params (robust_local) ---
             robust_method='trimmed',  # 'mcd' | 'trimmed' | 'huber' | 'none'
             support_fraction=0.75,    # MCD inlier fraction
             trim_fraction=0.1,        # trimmed discard fraction
             # --- Shared ---
             verbose=False,
             **kwargs)
```

When `method='standard'`, all robust-specific params are ignored and behavior is identical to today.

### Behavior by method

| `method` | `fit()` does | `transform(x)` returns |
|---|---|---|
| `'standard'` | sklearn PCA (unchanged) | PCA projection, shape (N, n_components) |
| `'robust_admm'` | Fixed-mu ALM solver -> L, S; reuses final SVD factors | Projection onto top-k singular vectors of L, shape (N, n_components) |
| `'robust_ialm'` | Increasing-mu ALM solver -> L, S; reuses final SVD factors | Projection onto top-k singular vectors of L, shape (N, n_components) |
| `'robust_local'` | Robust local PCA per-neighborhood + LTSA alignment | LTSA embedding, shape (N, n_components). Transductive — `transform(x_new)` raises `NotImplementedError` |

### Out-of-sample transform (global RPCA)

After `fit(D)` in robust global mode:
1. Decompose D = L + S
2. Reuse SVD factors from the final solver iteration (no redundant SVD)
3. Store `components_ = Vt[:n_components]` and `mean_ = L.mean(axis=0)`

**Centering uses the mean of L (the denoised component), not D.** This is intentional: D contains sparse corruption, and centering with corrupted data would inject corruption into the out-of-sample projection.

Then `transform(x_new)` centers using the fitted mean and projects onto `components_`. This makes global robust PCA inductive, just like standard PCA.

### fit_fraction interaction

When `fit_fraction < 1.0` and `method != 'standard'`, the robust solver operates on the subsetted data `D[:n_fit]`. The resulting L, S, and SVD factors all have shape `(n_fit, n_features)`. The stored `_fit_data` for `kernel_matrix()` is L (the denoised subset), not D.

### extra_outputs()

The `extra_outputs()` override **must call `super().extra_outputs()`** first (to preserve kernel_matrix, affinity_matrix collection from the base class), then add method-specific keys.

**Global RPCA** (`robust_admm` / `robust_ialm`):
- `low_rank_matrix`: full L matrix (m x n)
- `sparse_matrix`: full S matrix (m x n)
- `robust_rank`: estimated rank of L
- `convergence_history`: dict with keys `'error'`, `'rank'`, `'sparsity'` (lists per iteration)

**Local RPCA** (`robust_local`):
- `local_eigenvalues`: (n, d) per-point eigenspectra (descending order)
- `local_dims`: (n,) estimated local intrinsic dimension per point
- `local_variances`: (n,) total local variance per point (trace of local covariance)
- `outlier_masks`: (n, k) per-neighborhood outlier flags (bool, True = outlier)
- `condition_numbers`: (n,) condition number of each local covariance estimate
- `support_sizes`: (n,) number of points used in each local estimate

### kernel_matrix() / affinity_matrix()

When global robust, these operate on L (the denoised low-rank component) instead of D. When `robust_local`, default base class behavior (operates on fitted data).

## Solver Internals: `utils/robust_pca_solvers.py`

### Solver Naming Clarification

Both global solvers use the Augmented Lagrangian Method (ALM) formulation. The iteration body is identical — the only difference is the mu update rule:
- **ADMM** (`rpca_admm`): fixed mu throughout. Simpler, slower convergence.
- **IALM** (`rpca_ialm`): mu increases each iteration (`mu *= rho`). Faster convergence.

The "ADMM" name follows the convention used in the RPCA literature (Candes et al. 2011, dganguli), even though the formulation is technically fixed-mu ALM rather than classical ADMM with separate per-variable augmented Lagrangian terms.

### Global RPCA Public API

```python
class RobustPCAResult(NamedTuple):
    L: np.ndarray             # Low-rank component (m x n)
    S: np.ndarray             # Sparse component (m x n)
    rank: int                 # Estimated rank of L
    n_iter: int               # Number of iterations run
    convergence_history: dict # {'error': [...], 'rank': [...], 'sparsity': [...]}
                              # 'sparsity' = np.count_nonzero(S) / S.size (fraction of nonzero entries)
    svd_factors: tuple        # (U, sigma, Vt) from the final SVT step — avoids redundant SVD

def rpca_admm(D, lmbda=None, max_iter=500, tol=1e-7, delta=None,
              mu=None, use_truncated_svd=True, verbose=False) -> RobustPCAResult:
    """Fixed-mu ALM solver for Principal Component Pursuit."""

def rpca_ialm(D, lmbda=None, max_iter=100, tol=1e-7, delta=None,
              mu=None, mu_max=1e7, rho=1.5,
              use_truncated_svd=True, verbose=False) -> RobustPCAResult:
    """Inexact (increasing-mu) ALM solver for Principal Component Pursuit."""
```

### Algorithm (shared iteration body)

Both solvers use the same iteration:

```
Initialize:
  ADMM:  S_0 = 0, Y_0 = 0, mu = m*n / (4 * sum(|D|))
  IALM:  S_0 = 0, Y_0 = D / J(D), mu_0 = 1e-5
         where J(D) = max(sigma_max(D), ||D||_inf / lambda)
         sigma_max(D) = spectral norm = np.linalg.norm(D, ord=2)
         ||D||_inf    = matrix infinity norm (max row sum) = np.linalg.norm(D, ord=np.inf)

Repeat:
    L_{k+1} = SVT_{1/mu_k}(D - S_k + mu_k^{-1} Y_k)
    S_{k+1} = Shrink_{lambda/mu_k}(D - L_{k+1} + mu_k^{-1} Y_k)
    Y_{k+1} = Y_k + mu_k(D - L_{k+1} - S_{k+1})

    ADMM:  mu_{k+1} = mu_k                       (fixed)
    IALM:  mu_{k+1} = min(rho * mu_k, mu_max)    (increasing)

Until ||D - L - S||_F / ||D||_F < tol
```

### Stable PCP Modification

When `delta` is set, the dual update is modified with a projection step:

```
Y_{k+1} = Y_k + mu_k(D - L_{k+1} - S_{k+1})
residual = ||D - L_{k+1} - S_{k+1}||_F
if residual > delta:
    Y_{k+1} = Y_{k+1} * (delta / residual)   # project dual to respect noise bound
```

Convergence criterion changes to: `residual <= delta` (absolute, not relative).

### SVD Strategy (Critical for Performance)

Internal types:

```python
class SVTResult(NamedTuple):
    matrix: np.ndarray   # Thresholded low-rank approximation (m x n)
    U: np.ndarray        # Left singular vectors (thresholded subset)
    sigma: np.ndarray    # Singular values (after subtracting tau)
    Vt: np.ndarray       # Right singular vectors (thresholded subset)
```

Internal function `_svt(X, tau, prev_rank, rank_buffer=5, use_truncated=True)`:

```python
if use_truncated and prev_rank is not None and (prev_rank + rank_buffer) < min(m, n) / 2:
    # Try truncated SVD with k = prev_rank + buffer
    U, sigma, Vt = scipy.sparse.linalg.svds(X, k=prev_rank + rank_buffer)
    # svds returns ascending order — sort descending
    idx = np.argsort(sigma)[::-1]
    U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
    if sigma.min() > tau:
        # Missed components - fall back to full SVD
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
else:
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

# Threshold: keep only sigma_i > tau
mask = sigma > tau
if not mask.any():
    return SVTResult(matrix=np.zeros_like(X), U=U[:, :0], sigma=np.array([]), Vt=Vt[:0, :])

U_k, sigma_k, Vt_k = U[:, mask], sigma[mask] - tau, Vt[mask, :]
L = U_k @ np.diag(sigma_k) @ Vt_k
return SVTResult(matrix=L, U=U_k, sigma=sigma_k, Vt=Vt_k)
```

**Key:** `_svt` returns a `SVTResult(matrix, U, sigma, Vt)` namedtuple, not just the matrix. The solver caches the final iteration's SVD factors in `RobustPCAResult.svd_factors` to avoid a redundant O(mn * min(m,n)) SVD when extracting components for `transform()`.

### Soft Thresholding

```python
def _shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)
```

### Default Parameters (Global RPCA)

| Parameter | Default | Source |
|---|---|---|
| lambda | `1 / sqrt(max(m, n))` | Candes et al. 2011 (theoretically optimal) |
| ADMM mu | `m * n / (4 * np.sum(np.abs(D)))` | Element-wise L1 norm (sum of absolute values), NOT `np.linalg.norm(D, ord=1)` which is the matrix 1-norm (max column sum) |
| IALM mu_0 | `1e-5` | Lin, Chen & Ma 2010 |
| rho | `1.5` | Conservative (aggressive: 6) |
| mu_max | `1e7` | Standard |
| max_iter | 500 (ADMM), 100 (IALM) | IALM converges faster. When using `robust_admm`, consider increasing `solver_max_iter` to 500 |
| tol | `1e-7` | Standard |

### Numerical Stability Notes

- Cache `norm_D = np.linalg.norm(D, 'fro')` for relative error computation
- Guard against D = 0 (return zeros immediately)
- In SVT, handle the case where all singular values < tau (return zeros)
- No in-place operations on input D
- `scipy.sparse.linalg.svds` returns singular values in ascending order; sort descending before use. Note: `svds` can be numerically unstable for very small k (k=1, k=2); the full SVD fallback handles this.
- **dtype handling:** solvers internally upcast to `float64` for numerical stability. The convergence tolerance `1e-7` assumes float64 precision. Input data is cast back to its original dtype in the result.

---

# Part 3: Robust Local PCA

## Motivation

Local geometry estimation (LID, local tangent spaces, spectral gaps) depends on neighborhood quality. In practice, neighborhoods are contaminated by hub points, shortcut neighbors (manifold folding), noise, and DR distortion. Standard local PCA is not robust to any of these — a single outlier neighbor can dramatically rotate the estimated tangent space.

## Robust Local PCA Public API

In `utils/robust_pca_solvers.py`:

```python
class RobustLocalPCAResult(NamedTuple):
    local_bases: np.ndarray       # (n, n_components, d) local PC bases per point
    local_eigenvalues: np.ndarray # (n, d) full local eigenspectra (descending)
    local_dims: np.ndarray        # (n,) estimated local intrinsic dimension
    local_variances: np.ndarray   # (n,) total local variance (trace of local cov)
    outlier_masks: np.ndarray     # (n, k) bool, True = flagged as outlier
    condition_numbers: np.ndarray # (n,) condition number of local covariance
    support_sizes: np.ndarray     # (n,) points used per local estimate

def robust_local_pca(
    X,                          # (n, d) data matrix
    n_neighbors=20,             # k for k-NN
    n_components=None,          # local dim; if None, estimate per-point via eigenvalue gap
    robust_method='trimmed',    # 'mcd' | 'trimmed' | 'huber' | 'none'
    support_fraction=0.75,      # MCD inlier fraction
    trim_fraction=0.1,          # trimmed discard fraction
    precomputed_neighbors=None, # (n, k) index array
    precomputed_distances=None, # (n, k) distance array
    cache=None,                 # shared kNN cache (passed to compute_knn)
    random_state=42,
) -> RobustLocalPCAResult:
    """Robust local tangent space estimation per point."""
```

## Robust Covariance Methods

### Method 1: Trimmed (`robust_method='trimmed'`)

Simplest. Works in any dimension. Very fast.

1. Compute robust centroid (coordinate-wise median) of neighborhood
2. Compute distances from each neighbor to centroid
3. Remove `trim_fraction` furthest points
4. Compute empirical covariance on remaining points
5. Eigendecompose

### Method 2: MCD (`robust_method='mcd'`)

Uses `sklearn.covariance.MinCovDet` per neighborhood.

**Fallback:** When `k <= d` (common in high-dimensional data), MCD is undefined. Automatically falls back to `trimmed` with a logged warning.

### Method 3: Huber (`robust_method='huber'`)

Iteratively reweighted covariance with Huber weights:
1. Start with empirical covariance and mean
2. Compute Mahalanobis distances
3. Assign Huber weights: `w_i = min(1, c / d_i)` where `c = sqrt(chi2_ppf(0.95, d))`
4. Recompute weighted mean and covariance
5. Repeat until convergence (max 20 iterations, tol=1e-4)

**Fallback:** When `k <= d`, falls back to `trimmed` (can't invert covariance).

### Method 4: None (`robust_method='none'`)

Standard empirical covariance. Baseline for comparison.

## Local Dimension Estimation

When `n_components=None`, estimate local intrinsic dimension per-point via eigenvalue ratio gap:

```python
def _estimate_local_dim(eigenvalues):
    """Find largest gap in log-eigenvalue spectrum.
    Returns index of gap + 1 = estimated dimension."""
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) <= 1:
        return 1
    log_evals = np.log(pos)
    gaps = log_evals[:-1] - log_evals[1:]
    return int(np.argmax(gaps) + 1)
```

## LTSA Alignment (for embedding output)

After computing per-point robust local tangent bases `{(U_i, eigenvalues_i)}`, align them into a global embedding via Local Tangent Space Alignment (Zhang & Zha 2004):

1. For each point i, project its k neighbors onto the local tangent basis U_i (n_components top eigenvectors)
2. Build the alignment matrix B (n x n, sparse): for each neighborhood, compute the local reconstruction that maps local coordinates back to global indices
3. Eigendecompose B to get the d-dimensional embedding (smallest non-trivial eigenvectors)

This is the standard LTSA algorithm but with robust local tangent spaces instead of naive PCA tangent spaces.

**Implementation note:** Use `scipy.sparse.linalg.eigsh` on the alignment matrix (it's sparse and symmetric). The smallest `n_components + 1` eigenvalues correspond to the embedding; discard the trivial zero eigenvalue.

**Transductive:** LTSA is transductive. `transform(x_new)` raises `NotImplementedError` — same pattern as other transductive modules (e.g., Leiden, ReebGraph).

## kNN Integration

Uses `compute_knn()` from the new `utils/knn.py` (FAISS-accelerated, shared cache). Accepts `precomputed_neighbors` and `precomputed_distances` to skip recomputation when neighborhoods are already available from a previous step.

```python
if precomputed_neighbors is not None:
    indices, distances = precomputed_neighbors, precomputed_distances
else:
    distances, indices = compute_knn(X, k=n_neighbors, include_self=False, cache=cache)
```

## Parallelization

Start with a simple Python loop over points. Profile later — for small neighborhoods (k < 100), the per-point eigendecomposition is fast and joblib overhead may dominate. Add parallelization as a follow-up if profiling shows it's needed.

---

# Part 4: Hydra Configs

### `robust_pca.yaml` (new — global RPCA)

```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
neighborhood_size: ${neighborhood_size}
method: robust_ialm
lmbda: null
delta: null
solver_max_iter: 100
tol: 1.0e-7
mu: null
mu_max: 1.0e7
rho: 1.5
use_truncated_svd: true
verbose: false
```

### `robust_local_pca.yaml` (new — local RPCA)

```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
neighborhood_size: ${neighborhood_size}
method: robust_local
robust_method: trimmed
support_fraction: 0.75
trim_fraction: 0.1
verbose: false
```

### `pca.yaml` (unchanged)

```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
```

---

# Part 5: Test Plan

## `tests/test_robust_pca.py` (Global RPCA)

**Utility:**
```python
def make_rpca_test_data(m=200, n=100, rank=5, sparse_frac=0.05,
                         noise_std=0.0, seed=42):
    """Generate synthetic D = L + S (+ noise) with known ground truth."""
```

**Tests:**

1. **test_robust_pca_recovery_ialm** — IALM recovers L, S from noiseless D = L + S. Check `rel_err(L) < 0.01`, `rank == 5`, embedding shape `(200, 5)`.

2. **test_robust_pca_recovery_admm** — Same recovery test with ADMM solver.

3. **test_admm_ialm_agree** — Both solvers produce similar L, S on the same input. `rel_diff(L) < 0.05`.

4. **test_stable_pcp_with_noise** — Stable variant with `noise_std=0.1` and appropriate delta. Recovery within 10% relative error.

5. **test_convergence_history** — Error list is populated, final error < 1% of initial error.

6. **test_ialm_fewer_iterations_than_admm** — IALM converges in fewer iterations at the same tolerance. Uses larger matrix (500x200) for reliable convergence speed difference. Assertion: `ialm.n_iter <= admm.n_iter` (non-strict to avoid flakiness).

7. **test_standard_pca_unchanged** — `method='standard'` (default) produces identical results to current PCAModule. Backward compatibility.

8. **test_robust_pca_api_integration** — Callable via `manylatents.api.run()`:
   ```python
   from manylatents.api import run
   result = run(
       input_data=X.astype(np.float32),
       algorithms={'latent': {
           '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
           'n_components': 5,
           'method': 'robust_ialm',
       }}
   )
   assert result['embeddings'].shape == (200, 5)
   ```

9. **test_robust_pca_extra_outputs** — Verify `extra_outputs()` returns `low_rank_matrix`, `sparse_matrix`, `robust_rank`, `convergence_history`. Also verify that base-class outputs (kernel_matrix, affinity_matrix) are still present.

10. **test_robust_pca_transform_new_data** — After fitting, `transform(x_new)` works on unseen data with correct shape.

## `tests/test_robust_local_pca.py` (Local RPCA)

**Utility:**
```python
def make_robust_local_pca_test_data(n=1000, noise_std=0.0,
                                      contamination_frac=0.05, seed=42):
    """Swiss roll (intrinsic dim=2) with neighborhood contamination."""
```

**Tests:**

1. **test_robust_vs_naive_lid_accuracy** — Robust local PCA gives more accurate local dimension estimates (closer to true dim=2) than naive local PCA with 10% contamination. Uses `robust_method='trimmed'` vs `'none'`.

2. **test_methods_agree_on_clean_data** — Without contamination, all methods (`'none'`, `'trimmed'`, `'mcd'`, `'huber'`) produce similar local dimension estimates (>80% agreement).

3. **test_mcd_falls_back_when_k_less_than_d** — When `n_neighbors < n_features`, MCD falls back to trimmed without crashing.

4. **test_precomputed_neighbors** — Accepts precomputed neighborhood indices and distances.

5. **test_eigenvalue_spectra_shape** — `local_eigenvalues` shape is `(n, d)`, `local_bases` shape is `(n, n_components, d)`.

6. **test_robust_local_pca_api_integration** — Callable via `manylatents.api.run()`:
   ```python
   from manylatents.api import run
   from sklearn.datasets import make_swiss_roll
   X, _ = make_swiss_roll(500, random_state=42)
   result = run(
       input_data=X.astype(np.float32),
       algorithms={'latent': {
           '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
           'n_components': 2,
           'method': 'robust_local',
           'robust_method': 'trimmed',
       }}
   )
   assert result['embeddings'].shape == (500, 2)
   ```

7. **test_outlier_diagnostics** — `extra_outputs()` returns `outlier_masks`, `condition_numbers`, `support_sizes` with correct shapes.

8. **test_robust_local_pca_transform_raises** — `transform(x_new)` on unseen data raises `NotImplementedError` (transductive method).

9. **test_robust_local_pca_improves_lid** — Integration test: robust local eigenspectra yield LID estimates closer to true intrinsic dimension than naive, on contaminated Swiss roll.

## `tests/test_knn_refactor.py` (kNN backward compat)

1. **test_import_from_old_path** — `from manylatents.utils.metrics import compute_knn` still works.
2. **test_import_from_new_path** — `from manylatents.utils.knn import compute_knn` works.
3. **test_both_paths_same_function** — Both imports resolve to the same function object.

---

# Part 6: Future Extensions (Not In Scope)

- GPU backend via PyTorch (`torch.linalg.svd`, `torch.svd_lowrank`)
- Missing value support (masked entries)
- Streaming/online RPCA
- GoDec solver for very large matrices
- Kernel PCA, Sparse PCA as additional `method` values
- `explained_variance_ratio_` for robust mode
- Nystrom out-of-sample extension for robust_local LTSA
- Parallelized per-point loop (joblib) for robust_local_pca
- Tyler's M-estimator as additional `robust_method`
- Grassmann average (Hauberg et al. 2014) for robust subspace estimation

---

# References

1. Candes, E.J., Li, X., Ma, Y., & Wright, J. (2011). Robust Principal Component Analysis? JACM, 58(3).
2. Lin, Z., Chen, M., & Ma, Y. (2010). The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices. arXiv:1009.5055.
3. Zhou, Z., Li, X., Wright, J., Candes, E.J., & Ma, Y. (2010). Stable Principal Component Pursuit. ISIT 2010.
4. Zhang, Z. & Zha, H. (2004). Principal Manifolds and Nonlinear Dimensionality Reduction via Tangent Space Alignment. SIAM J. Sci. Comput. 26(1).
5. Rousseeuw, P.J. & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. Technometrics 41(3).
6. Hauberg, S., Feragen, A., & Black, M.J. (2014). Grassmann Averages for Scalable Robust PCA. CVPR 2014.
7. Levina, E. & Bickel, P.J. (2004). Maximum likelihood estimation of intrinsic dimension. NeurIPS 2004.
