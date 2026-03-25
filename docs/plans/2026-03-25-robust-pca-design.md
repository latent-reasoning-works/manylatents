# Robust PCA Design Spec

**Date:** 2026-03-25
**Status:** Reviewed

## Overview

Extend the existing `PCAModule` with a production-quality Robust PCA (RPCA) implementation. RPCA solves the Principal Component Pursuit (PCP) problem (Candes et al. 2011):

```
minimize ||L||_* + lambda ||S||_1
subject to D = L + S
```

where D is the observed data, L is low-rank, S is sparse. Default lambda = 1 / sqrt(max(m, n)).

Also supports the **stable variant** (Zhou et al. 2010) for noisy data:

```
minimize ||L||_* + lambda ||S||_1
subject to ||D - L - S||_F <= delta
```

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Integration point | Bake into existing `PCAModule` | One PCA class for all PCA-family operations |
| Method selection | `method` parameter: `'standard'`, `'robust_admm'`, `'robust_ialm'` | Extensible to future PCA variants |
| Solver location | `manylatents/utils/robust_pca_solvers.py` | Keeps `algorithms/latent/` for module classes; solvers are reusable utilities |
| transform() output | SVD projection of L into n_components dims | Natural DR interpretation: "PCA on denoised data" |
| Extra outputs | L, S, rank, convergence_history via `extra_outputs()` | Full decomposition accessible without changing the LatentModule contract |
| Stable PCP | Included (delta parameter) | Small addition to IALM; real data always has noise |
| Test strategy | Synthetic ground truth + API integration | No external baseline deps in CI |

## File Structure

```
manylatents/
  utils/
    robust_pca_solvers.py         # ADMM + IALM solver functions (public utility)
  algorithms/latent/
    pca.py                        # Extended PCAModule (method param, robust dispatch)
  configs/algorithms/latent/
    pca.yaml                      # Unchanged (method defaults to 'standard')
    robust_pca.yaml               # New: method='robust_ialm' + robust params
tests/
  test_robust_pca.py              # Ground truth recovery + API integration
```

No new packages, no new public exports in `algorithms/latent/__init__.py`.

**Note:** The existing `test_module_instantiation.py` auto-discovers all YAML configs in `configs/algorithms/latent/`. The `robust_pca.yaml` config will be auto-discovered, so the PCAModule constructor changes must be committed before the config file is added.

## API: PCAModule Changes

### Constructor

Current signature:
```python
def __init__(self, n_components=2, random_state=42, fit_fraction=1.0, **kwargs)
```

New signature (backward-compatible):
```python
def __init__(self, n_components=2, random_state=42, fit_fraction=1.0,
             method='standard',        # 'standard' | 'robust_admm' | 'robust_ialm'
             lmbda=None,               # lambda for RPCA (default: 1/sqrt(max(m,n)))
             solver_max_iter=500,      # max iterations for robust solver
             tol=1e-7,                 # convergence tolerance
             delta=None,              # stable PCP noise bound (None = exact PCP)
             mu=None,                 # initial penalty param (auto-set if None)
             mu_max=1e7,              # max penalty (IALM)
             rho=1.5,                # penalty growth rate (IALM)
             use_truncated_svd=True,   # adaptive truncated SVD for performance
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

### Out-of-sample transform

After `fit(D)` in robust mode:
1. Decompose D = L + S
2. Reuse SVD factors from the final solver iteration (no redundant SVD)
3. Store `components_ = Vt[:n_components]` and `mean_ = L.mean(axis=0)`

**Centering uses the mean of L (the denoised component), not D.** This is intentional: D contains sparse corruption, and centering with corrupted data would inject corruption into the out-of-sample projection.

Then `transform(x_new)` centers using the fitted mean and projects onto `components_`. This makes robust PCA inductive, just like standard PCA.

### fit_fraction interaction

When `fit_fraction < 1.0` and `method != 'standard'`, the robust solver operates on the subsetted data `D[:n_fit]`. The resulting L, S, and SVD factors all have shape `(n_fit, n_features)`. The stored `_fit_data` for `kernel_matrix()` is L (the denoised subset), not D.

### extra_outputs()

The `extra_outputs()` override **must call `super().extra_outputs()`** first (to preserve kernel_matrix, affinity_matrix collection from the base class), then add robust-specific keys.

When `method != 'standard'`, `extra_outputs()` adds:
- `low_rank_matrix`: full L matrix (m x n)
- `sparse_matrix`: full S matrix (m x n)
- `robust_rank`: estimated rank of L
- `convergence_history`: dict with keys `'error'`, `'rank'`, `'sparsity'` (lists per iteration)

### kernel_matrix() / affinity_matrix()

When robust, these operate on L (the denoised low-rank component) instead of D. This is the natural choice: the kernel of the denoised data.

## Solver Internals: `utils/robust_pca_solvers.py`

### Solver Naming Clarification

Both solvers use the Augmented Lagrangian Method (ALM) formulation. The iteration body is identical — the only difference is the mu update rule:
- **ADMM** (`rpca_admm`): fixed mu throughout. Simpler, slower convergence.
- **IALM** (`rpca_ialm`): mu increases each iteration (`mu *= rho`). Faster convergence.

The "ADMM" name follows the convention used in the RPCA literature (Candes et al. 2011, dganguli), even though the formulation is technically fixed-mu ALM rather than classical ADMM with separate per-variable augmented Lagrangian terms.

### Public API

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

### Default Parameters

| Parameter | Default | Source |
|---|---|---|
| lambda | `1 / sqrt(max(m, n))` | Candes et al. 2011 (theoretically optimal) |
| ADMM mu | `m * n / (4 * np.sum(np.abs(D)))` | Element-wise L1 norm (sum of absolute values), NOT `np.linalg.norm(D, ord=1)` which is the matrix 1-norm (max column sum) |
| IALM mu_0 | `1e-5` | Lin, Chen & Ma 2010 |
| rho | `1.5` | Conservative (aggressive: 6) |
| mu_max | `1e7` | Standard |
| max_iter | 500 (ADMM), 100 (IALM) | IALM converges faster |
| tol | `1e-7` | Standard |

### Numerical Stability Notes

- Cache `norm_D = np.linalg.norm(D, 'fro')` for relative error computation
- Guard against D = 0 (return zeros immediately)
- In SVT, handle the case where all singular values < tau (return zeros)
- No in-place operations on input D
- `scipy.sparse.linalg.svds` returns singular values in ascending order; sort descending before use. Note: `svds` can be numerically unstable for very small k (k=1, k=2); the full SVD fallback handles this.
- **dtype handling:** solvers internally upcast to `float64` for numerical stability. The convergence tolerance `1e-7` assumes float64 precision. Input data is cast back to its original dtype in the result.

## Hydra Config

### `robust_pca.yaml` (new)

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

### `pca.yaml` (unchanged)

```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
```

No changes needed — `method` defaults to `'standard'`.

## Test Plan

### `tests/test_robust_pca.py`

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

## Future Extensions (Not In Scope)

- GPU backend via PyTorch (`torch.linalg.svd`, `torch.svd_lowrank`)
- Missing value support (masked entries)
- Streaming/online variant
- GoDec solver for very large matrices
- Kernel PCA, Sparse PCA as additional `method` values
- `explained_variance_ratio_` for robust mode (fraction of nuclear norm by top-k components)

## References

1. Candes, E.J., Li, X., Ma, Y., & Wright, J. (2011). Robust Principal Component Analysis? JACM, 58(3).
2. Lin, Z., Chen, M., & Ma, Y. (2010). The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices. arXiv:1009.5055.
3. Zhou, Z., Li, X., Wright, J., Candes, E.J., & Ma, Y. (2010). Stable Principal Component Pursuit. ISIT 2010.
