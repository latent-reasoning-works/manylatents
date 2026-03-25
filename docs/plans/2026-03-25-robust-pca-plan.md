# Robust PCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PCAModule with global Robust PCA (ADMM/IALM), robust local PCA (MCD/trimmed/Huber + LTSA alignment), and refactor compute_knn into its own utils module.

**Architecture:** Three layers: (1) `utils/knn.py` extracts kNN infrastructure, (2) `utils/robust_pca_solvers.py` provides solver functions consumed by (3) `algorithms/latent/pca.py` which dispatches based on `method` param. All callable from `manylatents.api.run()`.

**Tech Stack:** NumPy, SciPy (sparse SVD, sparse eigsh), sklearn (MinCovDet, PCA), FAISS (kNN via existing infra)

**Spec:** `docs/plans/2026-03-25-robust-pca-design.md`

---

### Task 1: Extract compute_knn to utils/knn.py

**Files:**
- Create: `manylatents/utils/knn.py`
- Modify: `manylatents/utils/metrics.py:14-249`
- Create: `tests/test_knn_refactor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_knn_refactor.py`:

```python
"""Verify kNN refactor: both import paths work, same function object."""
import numpy as np


def test_import_from_new_path():
    from manylatents.utils.knn import compute_knn
    assert callable(compute_knn)


def test_import_from_old_path():
    from manylatents.utils.metrics import compute_knn
    assert callable(compute_knn)


def test_both_paths_same_function():
    from manylatents.utils.knn import compute_knn as knn_new
    from manylatents.utils.metrics import compute_knn as knn_old
    assert knn_new is knn_old


def test_content_key_same_function():
    from manylatents.utils.knn import _content_key as key_new
    from manylatents.utils.metrics import _content_key as key_old
    assert key_new is key_old


def test_compute_knn_basic():
    from manylatents.utils.knn import compute_knn
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5)).astype(np.float32)
    dists, idxs = compute_knn(X, k=5, include_self=False)
    assert dists.shape == (50, 5)
    assert idxs.shape == (50, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knn_refactor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'manylatents.utils.knn'`

- [ ] **Step 3: Create utils/knn.py with compute_knn and _content_key**

Create `manylatents/utils/knn.py`. Move `_content_key()` (lines 14-29 of `utils/metrics.py`) and `compute_knn()` (lines 148-249 of `utils/metrics.py`) into this new file. Keep all imports they need. The file should be self-contained:

```python
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _content_key(data) -> str:
    """O(1) content hash: shape + dtype + first/last row bytes."""
    import hashlib
    import torch
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.ascontiguousarray(data, dtype=np.float32)
    h = hashlib.sha256()
    h.update(f"{data.shape}{data.dtype}".encode())
    h.update(data[0].tobytes())
    h.update(data[-1].tobytes())
    return h.hexdigest()[:16]


def compute_knn(
    data: np.ndarray,
    k: int,
    include_self: bool = True,
    cache: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbors using FAISS-GPU > FAISS-CPU > sklearn.

    Args:
        data: (n_samples, n_features) float32 array.
        k: Number of neighbors (excluding self).
        include_self: If True, returns k+1 columns with self at index 0.
        cache: Optional dict for caching.

    Returns:
        (distances, indices) tuple.
    """
    # --- paste the full body from utils/metrics.py lines 174-249 verbatim ---
    # (import torch, check cache, FAISS/sklearn fallback, store cache, trim self)
    ...
```

Copy the **exact** body of `compute_knn` from `utils/metrics.py`. Do not modify any logic.

- [ ] **Step 4: Update utils/metrics.py — replace with re-exports**

In `manylatents/utils/metrics.py`:
- Remove the `_content_key` function (lines 14-29)
- Remove the `compute_knn` function (lines 148-249)
- Add at the top (after existing imports): `from manylatents.utils.knn import compute_knn, _content_key  # noqa: F401`
- Keep `_svd_gpu`, `_svd_cpu`, `compute_svd_cache`, `compute_eigenvalues`, `flatten_and_unroll_metrics`, and all other functions in place — they do NOT move.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_knn_refactor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All existing tests still pass (no import breakage)

- [ ] **Step 7: Commit**

```bash
git add manylatents/utils/knn.py manylatents/utils/metrics.py tests/test_knn_refactor.py
git commit -m "refactor: extract compute_knn to utils/knn.py with backward-compat re-export"
```

---

### Task 2: Global RPCA solver functions (_shrink, _svt, rpca_admm, rpca_ialm)

**Files:**
- Create: `manylatents/utils/robust_pca_solvers.py`
- Create: `tests/test_robust_pca.py` (solver-level tests first)

- [ ] **Step 1: Write solver-level failing tests**

Create `tests/test_robust_pca.py`:

```python
"""Tests for global Robust PCA solvers and PCAModule integration."""
import numpy as np
import pytest


def make_rpca_test_data(m=200, n=100, rank=5, sparse_frac=0.05,
                         noise_std=0.0, seed=42):
    """Generate synthetic D = L + S (+ noise) with known ground truth."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((n, rank))
    L_true = U @ V.T
    S_true = np.zeros((m, n))
    mask = rng.random((m, n)) < sparse_frac
    S_true[mask] = rng.uniform(-10, 10, size=mask.sum())
    noise = noise_std * rng.standard_normal((m, n)) if noise_std > 0 else 0
    D = L_true + S_true + noise
    return D, L_true, S_true


class TestSolverEdgeCases:
    def test_zero_matrix(self):
        from manylatents.utils.robust_pca_solvers import rpca_ialm
        D = np.zeros((50, 30))
        result = rpca_ialm(D)
        assert result.rank == 0
        assert np.allclose(result.L, 0)
        assert np.allclose(result.S, 0)


class TestSolverIALM:
    def test_recovery(self):
        from manylatents.utils.robust_pca_solvers import rpca_ialm
        D, L_true, S_true = make_rpca_test_data()
        result = rpca_ialm(D)
        rel_err = np.linalg.norm(result.L - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
        assert rel_err < 0.01, f"IALM L recovery error {rel_err:.4f} >= 0.01"
        assert result.rank == 5

    def test_convergence_history(self):
        from manylatents.utils.robust_pca_solvers import rpca_ialm
        D, _, _ = make_rpca_test_data()
        result = rpca_ialm(D)
        errors = result.convergence_history['error']
        assert len(errors) > 1
        assert errors[-1] < errors[0] * 0.01

    def test_svd_factors_returned(self):
        from manylatents.utils.robust_pca_solvers import rpca_ialm
        D, _, _ = make_rpca_test_data()
        result = rpca_ialm(D)
        U, sigma, Vt = result.svd_factors
        assert U.shape[1] == result.rank
        assert len(sigma) == result.rank
        assert Vt.shape[0] == result.rank


class TestSolverADMM:
    def test_recovery(self):
        from manylatents.utils.robust_pca_solvers import rpca_admm
        D, L_true, _ = make_rpca_test_data()
        result = rpca_admm(D)
        rel_err = np.linalg.norm(result.L - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
        assert rel_err < 0.01, f"ADMM L recovery error {rel_err:.4f} >= 0.01"


class TestSolverComparison:
    def test_admm_ialm_agree(self):
        from manylatents.utils.robust_pca_solvers import rpca_admm, rpca_ialm
        D, _, _ = make_rpca_test_data()
        r_admm = rpca_admm(D, tol=1e-7)
        r_ialm = rpca_ialm(D, tol=1e-7)
        rel_diff = np.linalg.norm(r_admm.L - r_ialm.L, 'fro') / np.linalg.norm(r_ialm.L, 'fro')
        assert rel_diff < 0.05

    def test_ialm_fewer_iterations(self):
        from manylatents.utils.robust_pca_solvers import rpca_admm, rpca_ialm
        D, _, _ = make_rpca_test_data(m=500, n=200)
        r_admm = rpca_admm(D, tol=1e-7)
        r_ialm = rpca_ialm(D, tol=1e-7)
        assert r_ialm.n_iter <= r_admm.n_iter


class TestStablePCP:
    def test_noisy_recovery(self):
        from manylatents.utils.robust_pca_solvers import rpca_ialm
        D, L_true, S_true = make_rpca_test_data(noise_std=0.1)
        noise_bound = 0.1 * np.sqrt(200 * 100)
        result = rpca_ialm(D, delta=noise_bound)
        rel_err = np.linalg.norm(result.L - L_true, 'fro') / np.linalg.norm(L_true, 'fro')
        assert rel_err < 0.1, f"Stable PCP L recovery error {rel_err:.4f} >= 0.1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robust_pca.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'manylatents.utils.robust_pca_solvers'`

- [ ] **Step 3: Implement _shrink and SVTResult**

Create `manylatents/utils/robust_pca_solvers.py`:

```python
"""Robust PCA solvers and robust local PCA utilities.

Provides:
- rpca_admm: Fixed-mu ALM for Principal Component Pursuit (D = L + S)
- rpca_ialm: Increasing-mu ALM (generally faster)
- robust_local_pca: Per-neighborhood robust tangent space estimation
"""
import logging
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SVTResult(NamedTuple):
    """Result of singular value thresholding."""
    matrix: np.ndarray
    U: np.ndarray
    sigma: np.ndarray
    Vt: np.ndarray


class RobustPCAResult(NamedTuple):
    """Result of global Robust PCA decomposition."""
    L: np.ndarray
    S: np.ndarray
    rank: int
    n_iter: int
    convergence_history: dict
    svd_factors: tuple


def _shrink(X: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft thresholding: sign(X) * max(|X| - tau, 0)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _svt(X: np.ndarray, tau: float, prev_rank: Optional[int] = None,
         rank_buffer: int = 5, use_truncated: bool = True) -> SVTResult:
    """Singular value thresholding with adaptive truncated SVD.

    Returns SVTResult(matrix, U, sigma, Vt) where sigma values are
    post-thresholding (sigma_i - tau for kept components).
    """
    m, n = X.shape
    min_dim = min(m, n)

    if (use_truncated and prev_rank is not None
            and (prev_rank + rank_buffer) < min_dim / 2
            and (prev_rank + rank_buffer) >= 1):
        from scipy.sparse.linalg import svds
        k = prev_rank + rank_buffer
        try:
            U, sigma, Vt = svds(X, k=k)
            # svds returns ascending order — sort descending
            idx = np.argsort(sigma)[::-1]
            U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
            if sigma[-1] > tau:
                # May have missed components — fall back to full SVD
                U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
        except Exception:
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

    mask = sigma > tau
    if not mask.any():
        return SVTResult(
            matrix=np.zeros_like(X),
            U=U[:, :0], sigma=np.array([]), Vt=Vt[:0, :]
        )

    U_k = U[:, mask]
    sigma_k = sigma[mask] - tau
    Vt_k = Vt[mask, :]
    L = (U_k * sigma_k[np.newaxis, :]) @ Vt_k  # avoid diag allocation
    return SVTResult(matrix=L, U=U_k, sigma=sigma_k, Vt=Vt_k)
```

- [ ] **Step 4: Implement rpca_ialm**

Add to `manylatents/utils/robust_pca_solvers.py`:

```python
def rpca_ialm(D: np.ndarray, lmbda: Optional[float] = None,
              max_iter: int = 100, tol: float = 1e-7,
              delta: Optional[float] = None,
              mu: Optional[float] = None, mu_max: float = 1e7,
              rho: float = 1.5, use_truncated_svd: bool = True,
              verbose: bool = False) -> RobustPCAResult:
    """Inexact ALM solver for Principal Component Pursuit.

    Solves: minimize ||L||_* + lambda ||S||_1  subject to D = L + S
    When delta is set, solves the stable variant: ||D - L - S||_F <= delta.
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(m, n))

    norm_D = np.linalg.norm(D, 'fro')
    if norm_D == 0:
        return RobustPCAResult(
            L=np.zeros_like(D), S=np.zeros_like(D),
            rank=0, n_iter=0,
            convergence_history={'error': [], 'rank': [], 'sparsity': []},
            svd_factors=(np.empty((m, 0)), np.array([]), np.empty((0, n)))
        )

    # Initialize: Y_0 = D / J(D)
    spectral_norm = np.linalg.norm(D, ord=2)
    inf_norm = np.linalg.norm(D, ord=np.inf)
    J_D = max(spectral_norm, inf_norm / lmbda)
    Y = D / J_D

    if mu is None:
        mu = 1e-5

    S = np.zeros_like(D)
    prev_rank = None
    history = {'error': [], 'rank': [], 'sparsity': []}

    for iteration in range(max_iter):
        # L step: SVT_{1/mu}(D - S + Y/mu)
        svt_result = _svt(D - S + Y / mu, 1.0 / mu,
                          prev_rank=prev_rank, use_truncated=use_truncated_svd)
        L = svt_result.matrix
        cur_rank = len(svt_result.sigma)
        prev_rank = cur_rank

        # S step: shrink_{lambda/mu}(D - L + Y/mu)
        S = _shrink(D - L + Y / mu, lmbda / mu)

        # Dual update
        residual_matrix = D - L - S
        residual_norm = np.linalg.norm(residual_matrix, 'fro')
        Y = Y + mu * residual_matrix

        # Stable PCP: project dual
        if delta is not None and residual_norm > delta:
            Y = Y * (delta / residual_norm)

        # Convergence tracking
        if delta is not None:
            error = residual_norm
            converged = residual_norm <= delta
        else:
            error = residual_norm / norm_D
            converged = error < tol

        sparsity = np.count_nonzero(S) / S.size
        history['error'].append(error)
        history['rank'].append(cur_rank)
        history['sparsity'].append(sparsity)

        if verbose:
            logger.info(f"IALM iter {iteration}: error={error:.2e}, rank={cur_rank}, "
                        f"sparsity={sparsity:.4f}, mu={mu:.2e}")

        if converged:
            break

        # Update mu
        mu = min(rho * mu, mu_max)

    return RobustPCAResult(
        L=L, S=S, rank=cur_rank, n_iter=iteration + 1,
        convergence_history=history,
        svd_factors=(svt_result.U, svt_result.sigma, svt_result.Vt)
    )
```

- [ ] **Step 5: Implement rpca_admm**

Add to `manylatents/utils/robust_pca_solvers.py`:

```python
def rpca_admm(D: np.ndarray, lmbda: Optional[float] = None,
              max_iter: int = 500, tol: float = 1e-7,
              delta: Optional[float] = None,
              mu: Optional[float] = None,
              use_truncated_svd: bool = True,
              verbose: bool = False) -> RobustPCAResult:
    """Fixed-mu ALM solver for Principal Component Pursuit.

    Same algorithm as rpca_ialm but mu stays fixed throughout.
    Generally slower convergence — prefer rpca_ialm.
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(m, n))

    norm_D = np.linalg.norm(D, 'fro')
    if norm_D == 0:
        return RobustPCAResult(
            L=np.zeros_like(D), S=np.zeros_like(D),
            rank=0, n_iter=0,
            convergence_history={'error': [], 'rank': [], 'sparsity': []},
            svd_factors=(np.empty((m, 0)), np.array([]), np.empty((0, n)))
        )

    if mu is None:
        mu = m * n / (4.0 * np.sum(np.abs(D)))

    Y = np.zeros_like(D)
    S = np.zeros_like(D)
    prev_rank = None
    history = {'error': [], 'rank': [], 'sparsity': []}

    for iteration in range(max_iter):
        svt_result = _svt(D - S + Y / mu, 1.0 / mu,
                          prev_rank=prev_rank, use_truncated=use_truncated_svd)
        L = svt_result.matrix
        cur_rank = len(svt_result.sigma)
        prev_rank = cur_rank

        S = _shrink(D - L + Y / mu, lmbda / mu)

        residual_matrix = D - L - S
        residual_norm = np.linalg.norm(residual_matrix, 'fro')
        Y = Y + mu * residual_matrix

        if delta is not None and residual_norm > delta:
            Y = Y * (delta / residual_norm)

        if delta is not None:
            error = residual_norm
            converged = residual_norm <= delta
        else:
            error = residual_norm / norm_D
            converged = error < tol

        sparsity = np.count_nonzero(S) / S.size
        history['error'].append(error)
        history['rank'].append(cur_rank)
        history['sparsity'].append(sparsity)

        if verbose:
            logger.info(f"ADMM iter {iteration}: error={error:.2e}, rank={cur_rank}, "
                        f"sparsity={sparsity:.4f}")

        if converged:
            break

    return RobustPCAResult(
        L=L, S=S, rank=cur_rank, n_iter=iteration + 1,
        convergence_history=history,
        svd_factors=(svt_result.U, svt_result.sigma, svt_result.Vt)
    )
```

- [ ] **Step 6: Run solver tests**

Run: `uv run pytest tests/test_robust_pca.py -v`
Expected: All 7 solver tests PASS

- [ ] **Step 7: Commit**

```bash
git add manylatents/utils/robust_pca_solvers.py tests/test_robust_pca.py
git commit -m "feat: add global RPCA solvers (ADMM + IALM) with tests"
```

---

### Task 3: Extend PCAModule with global RPCA methods

**Files:**
- Modify: `manylatents/algorithms/latent/pca.py`
- Add tests to: `tests/test_robust_pca.py`

- [ ] **Step 1: Write PCAModule-level failing tests**

Append to `tests/test_robust_pca.py`:

```python
class TestPCAModuleRobust:
    def test_standard_pca_unchanged(self):
        """Default method='standard' produces identical results."""
        from manylatents.algorithms.latent.pca import PCAModule
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10)).astype(np.float32)

        mod_old = PCAModule(n_components=2)
        emb_old = mod_old.fit_transform(X)

        mod_new = PCAModule(n_components=2, method='standard')
        emb_new = mod_new.fit_transform(X)

        np.testing.assert_array_equal(emb_old, emb_new)

    def test_robust_ialm_embedding_shape(self):
        from manylatents.algorithms.latent.pca import PCAModule
        D, _, _ = make_rpca_test_data()
        mod = PCAModule(n_components=5, method='robust_ialm')
        emb = mod.fit_transform(D.astype(np.float32))
        assert emb.shape == (200, 5)

    def test_robust_ialm_extra_outputs(self):
        from manylatents.algorithms.latent.pca import PCAModule
        D, L_true, _ = make_rpca_test_data()
        mod = PCAModule(n_components=5, method='robust_ialm')
        mod.fit(D.astype(np.float32))
        extras = mod.extra_outputs()
        assert 'low_rank_matrix' in extras
        assert 'sparse_matrix' in extras
        assert 'robust_rank' in extras
        assert 'convergence_history' in extras
        assert extras['robust_rank'] == 5
        # Base-class outputs should still be present
        assert 'kernel_matrix' in extras

    def test_robust_transform_new_data(self):
        from manylatents.algorithms.latent.pca import PCAModule
        D, _, _ = make_rpca_test_data()
        mod = PCAModule(n_components=5, method='robust_ialm')
        mod.fit(D.astype(np.float32))
        rng = np.random.default_rng(99)
        X_new = rng.standard_normal((50, 100)).astype(np.float32)
        emb = mod.transform(X_new)
        assert emb.shape == (50, 5)

    def test_fit_fraction_interaction(self):
        from manylatents.algorithms.latent.pca import PCAModule
        D, _, _ = make_rpca_test_data(m=200, n=100)
        mod = PCAModule(n_components=5, method='robust_ialm', fit_fraction=0.5)
        mod.fit(D.astype(np.float32))
        # L/S should have shape of the subsetted data
        extras = mod.extra_outputs()
        assert extras['low_rank_matrix'].shape == (100, 100)  # 50% of 200
        # transform on new data should still work
        emb = mod.transform(D.astype(np.float32))
        assert emb.shape == (200, 5)

    def test_robust_pca_api_integration(self):
        from manylatents.api import run
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 50)).astype(np.float32)
        result = run(
            input_data=X,
            algorithms={'latent': {
                '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                'n_components': 5,
                'method': 'robust_ialm',
            }}
        )
        assert result['embeddings'].shape == (200, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_robust_pca.py::TestPCAModuleRobust -v`
Expected: FAIL — `TypeError: PCAModule.__init__() got an unexpected keyword argument 'method'`

- [ ] **Step 3: Extend PCAModule constructor and fit/transform**

Replace the full contents of `manylatents/algorithms/latent/pca.py`. Key changes:
- Add `method`, global RPCA params, local RPCA params, `verbose` to `__init__`
- `fit()` dispatches to `_fit_standard()` or `_fit_robust_global()` based on `self.method`
- `transform()` dispatches similarly
- Override `fit_transform()` for non-standard methods
- Store `_robust_result`, `_components`, `_mean` for robust modes
- Override `extra_outputs()` calling `super().extra_outputs()` first
- Override `kernel_matrix()` / `affinity_matrix()` for robust global modes

```python
import logging

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from .latent_module_base import LatentModule, _to_numpy, _to_output

logger = logging.getLogger(__name__)


class PCAModule(LatentModule):
    def __init__(self,
                 n_components: int = 2,
                 random_state: int = 42,
                 fit_fraction: float = 1.0,
                 method: str = 'standard',
                 # Global RPCA params
                 lmbda=None,
                 solver_max_iter: int = 500,
                 tol: float = 1e-7,
                 delta=None,
                 mu=None,
                 mu_max: float = 1e7,
                 rho: float = 1.5,
                 use_truncated_svd: bool = True,
                 # Local RPCA params
                 n_neighbors: int = 20,
                 robust_method: str = 'trimmed',  # 'mcd'|'trimmed'|'huber'|'gaussian'|'none'
                 support_fraction: float = 0.75,
                 trim_fraction: float = 0.1,
                 # Shared
                 verbose: bool = False,
                 **kwargs):
        super().__init__(n_components=n_components,
                         init_seed=random_state,
                         **kwargs)
        self.method = method
        self.fit_fraction = fit_fraction
        self.random_state = random_state
        self.verbose = verbose

        # Global RPCA params
        self.lmbda = lmbda
        self.solver_max_iter = solver_max_iter
        self.tol = tol
        self.delta = delta
        self.mu = mu
        self.mu_max = mu_max
        self.rho = rho
        self.use_truncated_svd = use_truncated_svd

        # Local RPCA params
        self.n_neighbors = (self.neighborhood_size
                            if self.neighborhood_size is not None
                            else n_neighbors)
        self.robust_method = robust_method
        self.support_fraction = support_fraction
        self.trim_fraction = trim_fraction

        # Standard PCA model (only used for method='standard')
        if self.method == 'standard':
            self.model = PCA(n_components=n_components,
                             random_state=random_state)

        self._is_fitted = False
        self._fit_data = None
        self._robust_result = None
        self._components = None
        self._mean = None
        self._embedding = None

    def fit(self, x, y=None) -> None:
        x_np = _to_numpy(x)
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        x_fit = x_np[:n_fit]

        if self.method == 'standard':
            self._fit_standard(x_fit)
        elif self.method in ('robust_admm', 'robust_ialm'):
            self._fit_robust_global(x_fit)
        elif self.method == 'robust_local':
            self._fit_robust_local(x_fit)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        self._is_fitted = True

    def _fit_standard(self, x_np):
        self.model.fit(x_np)
        self._fit_data = x_np

    def _fit_robust_global(self, x_np):
        from manylatents.utils.robust_pca_solvers import rpca_admm, rpca_ialm
        solver = rpca_ialm if self.method == 'robust_ialm' else rpca_admm
        max_iter = self.solver_max_iter
        kwargs = dict(lmbda=self.lmbda, max_iter=max_iter, tol=self.tol,
                      delta=self.delta, mu=self.mu,
                      use_truncated_svd=self.use_truncated_svd,
                      verbose=self.verbose)
        if self.method == 'robust_ialm':
            kwargs['mu_max'] = self.mu_max
            kwargs['rho'] = self.rho
        result = solver(x_np, **kwargs)
        self._robust_result = result

        # Extract components from SVD factors (no redundant SVD)
        U, sigma, Vt = result.svd_factors
        nc = min(self.n_components, len(sigma))
        self._components = Vt[:nc]
        self._mean = result.L.mean(axis=0)
        # Store L as fit_data for kernel_matrix
        self._fit_data = result.L

    def _fit_robust_local(self, x_np):
        # Placeholder — implemented in Task 5
        raise NotImplementedError("robust_local not yet implemented")

    def transform(self, x):
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")

        x_np = _to_numpy(x)

        if self.method == 'standard':
            embedding = self.model.transform(x_np)
        elif self.method in ('robust_admm', 'robust_ialm'):
            embedding = (x_np - self._mean) @ self._components.T
        elif self.method == 'robust_local':
            if tuple(x_np.shape) == getattr(self, '_X_fit_shape', None):
                return _to_output(self._embedding, x)
            raise NotImplementedError(
                "Out-of-sample transform not implemented for robust_local. "
                "Use fit_transform() instead."
            )
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return _to_output(embedding, x)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        if self.method == 'robust_local':
            return _to_output(self._embedding, x)
        return self.transform(x)

    def extra_outputs(self) -> dict:
        extras = super().extra_outputs()

        if self.method in ('robust_admm', 'robust_ialm') and self._robust_result is not None:
            r = self._robust_result
            extras['low_rank_matrix'] = r.L
            extras['sparse_matrix'] = r.S
            extras['robust_rank'] = r.rank
            extras['convergence_history'] = r.convergence_history

        return extras

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PCA model is not fitted yet. Call `fit` first.")
        if self.method == 'robust_local':
            raise NotImplementedError(
                "kernel_matrix not available for robust_local method."
            )

        if self.method == 'standard':
            X_centered = self._fit_data - self.model.mean_
        else:
            X_centered = self._fit_data - self._mean

        K = X_centered @ X_centered.T
        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K

    def affinity_matrix(self, ignore_diagonal: bool = False,
                        use_symmetric: bool = False) -> np.ndarray:
        K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
        n = self._fit_data.shape[0]
        return K / (n - 1)
```

- [ ] **Step 4: Run PCAModule tests**

Run: `uv run pytest tests/test_robust_pca.py -v`
Expected: All tests PASS (both solver-level and module-level)

- [ ] **Step 5: Run full test suite for regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All existing tests pass — PCAModule backward compat preserved

- [ ] **Step 6: Commit**

```bash
git add manylatents/algorithms/latent/pca.py tests/test_robust_pca.py
git commit -m "feat: extend PCAModule with global RPCA methods (admm, ialm)"
```

---

### Task 4: Hydra config for global RPCA

**Files:**
- Create: `manylatents/configs/algorithms/latent/robust_pca.yaml`

- [ ] **Step 1: Create the config**

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

- [ ] **Step 2: Verify auto-discovery works**

Run: `uv run pytest tests/test_module_instantiation.py -v -k robust_pca`
Expected: The new YAML is auto-discovered and PCAModule instantiates without error

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add manylatents/configs/algorithms/latent/robust_pca.yaml
git commit -m "feat: add robust_pca.yaml Hydra config (method=robust_ialm)"
```

---

### Task 5: Robust local PCA solver functions

**Files:**
- Modify: `manylatents/utils/robust_pca_solvers.py` (add robust_local_pca, covariance methods, LTSA)
- Create: `tests/test_robust_local_pca.py`

- [ ] **Step 1: Write failing tests for robust_local_pca utility**

Create `tests/test_robust_local_pca.py`:

```python
"""Tests for robust local PCA and PCAModule robust_local integration."""
import numpy as np
import pytest


def make_robust_local_pca_test_data(n=500, noise_std=0.0,
                                      contamination_frac=0.05, seed=42):
    """Swiss roll (intrinsic dim=2) with neighborhood contamination."""
    from sklearn.datasets import make_swiss_roll
    rng = np.random.default_rng(seed)
    X, t = make_swiss_roll(n, noise=noise_std, random_state=seed)
    n_contaminated = int(n * contamination_frac)
    contam_idx = rng.choice(n, n_contaminated, replace=False)
    X[contam_idx] += rng.normal(0, 5, size=(n_contaminated, 3))
    return X.astype(np.float32), t, contam_idx


class TestRobustLocalPCASolver:
    def test_trimmed_basic_shape(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        result = robust_local_pca(X, n_neighbors=15, n_components=2,
                                   robust_method='trimmed')
        assert result.local_bases.shape == (100, 2, 5)
        assert result.local_eigenvalues.shape[0] == 100
        assert result.local_dims.shape == (100,)
        assert np.all(result.local_dims == 2)
        assert result.outlier_masks.shape == (100, 15)

    def test_none_method_baseline(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        X, _, _ = make_robust_local_pca_test_data(contamination_frac=0.0)
        result = robust_local_pca(X, n_neighbors=20, n_components=2,
                                   robust_method='none')
        assert result.local_bases.shape[0] == len(X)

    def test_mcd_fallback_when_k_less_than_d(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 50)).astype(np.float32)
        # k=20 < d=50 — MCD should fall back to trimmed
        result = robust_local_pca(X, n_neighbors=20, robust_method='mcd')
        assert result.local_bases.shape[0] == 100

    def test_huber_runs(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5)).astype(np.float32)
        result = robust_local_pca(X, n_neighbors=20, n_components=2,
                                   robust_method='huber')
        assert result.local_bases.shape == (80, 2, 5)

    def test_gaussian_weighted_runs(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5)).astype(np.float32)
        result = robust_local_pca(X, n_neighbors=20, n_components=2,
                                   robust_method='gaussian')
        assert result.local_bases.shape == (80, 2, 5)

    def test_methods_agree_on_clean_data(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        X, _, _ = make_robust_local_pca_test_data(contamination_frac=0.0,
                                                    noise_std=0.1, n=300)
        results = {}
        for method in ['none', 'trimmed', 'mcd', 'huber', 'gaussian']:
            results[method] = robust_local_pca(X, n_neighbors=30,
                                                n_components=2,
                                                robust_method=method)
        for method in ['trimmed', 'mcd', 'huber', 'gaussian']:
            agreement = np.mean(
                results[method].local_dims == results['none'].local_dims
            )
            assert agreement > 0.7, f"{method} disagrees on {1-agreement:.0%}"

    def test_precomputed_neighbors(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        from manylatents.utils.knn import compute_knn
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        dists, indices = compute_knn(X, k=20, include_self=False)
        result = robust_local_pca(X, precomputed_neighbors=indices,
                                   precomputed_distances=dists,
                                   robust_method='trimmed', n_components=2)
        assert result.local_dims.shape == (100,)

    def test_robust_vs_naive_lid(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        X, _, contam_idx = make_robust_local_pca_test_data(
            n=1000, contamination_frac=0.1
        )
        robust = robust_local_pca(X, n_neighbors=30, robust_method='trimmed')
        naive = robust_local_pca(X, n_neighbors=30, robust_method='none')
        clean = np.ones(len(X), dtype=bool)
        clean[contam_idx] = False
        robust_median = np.median(robust.local_dims[clean])
        naive_median = np.median(naive.local_dims[clean])
        assert abs(robust_median - 2.0) <= abs(naive_median - 2.0) + 0.5

    def test_auto_dim_estimation(self):
        from manylatents.utils.robust_pca_solvers import robust_local_pca
        X, _, _ = make_robust_local_pca_test_data(n=500, contamination_frac=0.0)
        result = robust_local_pca(X, n_neighbors=30, n_components=None,
                                   robust_method='trimmed')
        # Most points on Swiss roll should estimate dim=2
        assert np.median(result.local_dims) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_robust_local_pca.py::TestRobustLocalPCASolver -v`
Expected: FAIL — `ImportError: cannot import name 'robust_local_pca'`

- [ ] **Step 3: Implement robust covariance methods**

Add to `manylatents/utils/robust_pca_solvers.py`:

```python
class RobustLocalPCAResult(NamedTuple):
    """Result of robust local PCA per point."""
    local_bases: np.ndarray
    local_eigenvalues: np.ndarray
    local_dims: np.ndarray
    local_variances: np.ndarray
    outlier_masks: np.ndarray
    condition_numbers: np.ndarray
    support_sizes: np.ndarray


def _estimate_local_dim(eigenvalues: np.ndarray) -> int:
    """Estimate local intrinsic dimension via largest log-eigenvalue gap."""
    pos = eigenvalues[eigenvalues > 1e-10]
    if len(pos) <= 1:
        return 1
    log_evals = np.log(pos)
    gaps = log_evals[:-1] - log_evals[1:]
    return int(np.argmax(gaps) + 1)


def _local_cov_none(points: np.ndarray):
    """Standard empirical covariance (no robustness)."""
    k = len(points)
    centered = points - points.mean(axis=0)
    cov = (centered.T @ centered) / max(k - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    outlier_mask = np.zeros(k, dtype=bool)
    return eigenvalues, eigenvectors, outlier_mask, k


def _local_cov_trimmed(points: np.ndarray, trim_fraction: float):
    """Trimmed covariance: remove furthest points from median centroid."""
    k, d = points.shape
    n_trim = int(np.floor(k * trim_fraction))
    if n_trim == 0:
        return _local_cov_none(points)
    centroid = np.median(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    keep_idx = np.argsort(dists)[:k - n_trim]
    trimmed = points[keep_idx]
    centered = trimmed - trimmed.mean(axis=0)
    n_kept = len(trimmed)
    cov = (centered.T @ centered) / max(n_kept - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    outlier_mask = np.ones(k, dtype=bool)
    outlier_mask[keep_idx] = False
    return eigenvalues, eigenvectors, outlier_mask, n_kept


def _local_cov_mcd(points: np.ndarray, support_fraction: float,
                   random_state: int, trim_fraction: float):
    """MCD covariance. Falls back to trimmed when k <= d."""
    k, d = points.shape
    if k <= d:
        logger.debug("MCD: k=%d <= d=%d, falling back to trimmed", k, d)
        return _local_cov_trimmed(points, trim_fraction)
    from sklearn.covariance import MinCovDet
    mcd = MinCovDet(support_fraction=support_fraction,
                    random_state=random_state)
    mcd.fit(points)
    eigenvalues, eigenvectors = np.linalg.eigh(mcd.covariance_)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    outlier_mask = ~mcd.support_
    return eigenvalues, eigenvectors, outlier_mask, int(mcd.support_.sum())


def _local_cov_gaussian(points: np.ndarray, distances: np.ndarray):
    """Gaussian kernel-weighted local covariance (AdaL-PCA style).

    Weights each neighbor by exp(-dist^2 / epsilon) where epsilon is
    the median squared distance (adaptive bandwidth).
    Ref: Mez et al., AdaL-PCA (github.com/LydiaMez/AdaL-PCA).
    """
    k, d = points.shape
    # Adaptive bandwidth: median of squared distances
    sq_dists = distances ** 2
    epsilon = max(np.median(sq_dists), 1e-10)
    weights = np.exp(-sq_dists / epsilon)
    weights /= max(weights.sum(), 1e-10)

    mu = np.average(points, weights=weights, axis=0)
    diff = points - mu
    W = np.diag(weights)
    cov = diff.T @ W @ diff  # weighted covariance

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    # No hard outlier rejection — all points contribute via soft weights
    outlier_mask = weights < (1.0 / k * 0.1)  # flag very low-weight points
    return eigenvalues, eigenvectors, outlier_mask, int((~outlier_mask).sum())


def _local_cov_huber(points: np.ndarray, trim_fraction: float,
                     max_iter: int = 20, tol: float = 1e-4):
    """Iteratively reweighted covariance with Huber weights."""
    from scipy.stats import chi2
    k, d = points.shape
    if k <= d:
        logger.debug("Huber: k=%d <= d=%d, falling back to trimmed", k, d)
        return _local_cov_trimmed(points, trim_fraction)

    c = np.sqrt(chi2.ppf(0.95, df=d))
    mu = points.mean(axis=0)
    centered = points - mu
    Sigma = (centered.T @ centered) / max(k - 1, 1)

    for _ in range(max_iter):
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)
        diff = points - mu
        mahal = np.sqrt(np.maximum(np.sum((diff @ Sigma_inv) * diff, axis=1), 0))
        weights = np.minimum(1.0, c / np.maximum(mahal, 1e-10))
        mu_new = np.average(points, weights=weights, axis=0)
        diff = points - mu_new
        W = np.diag(weights)
        w_sum = weights.sum()
        Sigma_new = (diff.T @ W @ diff) / max(w_sum, 1e-10)
        delta = np.linalg.norm(Sigma_new - Sigma, 'fro') / max(np.linalg.norm(Sigma, 'fro'), 1e-10)
        mu, Sigma = mu_new, Sigma_new
        if delta < tol:
            break

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    outlier_mask = weights < 0.5
    return eigenvalues, eigenvectors, outlier_mask, int((weights >= 0.5).sum())
```

- [ ] **Step 4: Implement robust_local_pca orchestrator**

Add to `manylatents/utils/robust_pca_solvers.py`:

```python
def robust_local_pca(
    X: np.ndarray,
    n_neighbors: int = 20,
    n_components: Optional[int] = None,
    robust_method: str = 'trimmed',
    support_fraction: float = 0.75,
    trim_fraction: float = 0.1,
    precomputed_neighbors: Optional[np.ndarray] = None,
    precomputed_distances: Optional[np.ndarray] = None,
    cache: Optional[dict] = None,
    random_state: int = 42,
) -> RobustLocalPCAResult:
    """Robust local tangent space estimation per point."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Get neighborhoods
    if precomputed_neighbors is not None:
        indices = precomputed_neighbors
        k = indices.shape[1]
    else:
        from manylatents.utils.knn import compute_knn
        dists, indices = compute_knn(X.astype(np.float32), k=n_neighbors,
                                      include_self=False, cache=cache)
        k = n_neighbors

    eig_dim = min(k, d)
    all_eigenvalues = np.zeros((n, eig_dim))
    all_eigenvectors = []
    all_dims = np.zeros(n, dtype=int)
    all_variances = np.zeros(n)
    all_outlier_masks = np.zeros((n, k), dtype=bool)
    all_condition = np.zeros(n)
    all_support = np.zeros(n, dtype=int)

    for i in range(n):
        nbr_points = X[indices[i]]

        if robust_method == 'none':
            evals, evecs, omask, sup = _local_cov_none(nbr_points)
        elif robust_method == 'trimmed':
            evals, evecs, omask, sup = _local_cov_trimmed(nbr_points, trim_fraction)
        elif robust_method == 'mcd':
            evals, evecs, omask, sup = _local_cov_mcd(
                nbr_points, support_fraction, random_state, trim_fraction)
        elif robust_method == 'huber':
            evals, evecs, omask, sup = _local_cov_huber(nbr_points, trim_fraction)
        elif robust_method == 'gaussian':
            nbr_dists = (precomputed_distances[i] if precomputed_distances is not None
                         else np.linalg.norm(nbr_points - X[i], axis=1))
            evals, evecs, omask, sup = _local_cov_gaussian(nbr_points, nbr_dists)
        else:
            raise ValueError(f"Unknown robust_method: {robust_method!r}")

        # Clamp negative eigenvalues (numerical noise)
        evals = np.maximum(evals, 0.0)

        # Store (pad/truncate to eig_dim)
        n_evals = min(len(evals), eig_dim)
        all_eigenvalues[i, :n_evals] = evals[:n_evals]
        all_eigenvectors.append(evecs)
        all_variances[i] = evals.sum()
        all_outlier_masks[i] = omask
        all_support[i] = sup

        pos_evals = evals[evals > 1e-10]
        if len(pos_evals) > 1:
            all_condition[i] = pos_evals[0] / pos_evals[-1]
        else:
            all_condition[i] = 1.0

        if n_components is not None:
            all_dims[i] = n_components
        else:
            all_dims[i] = _estimate_local_dim(evals)

    # Build local_bases array
    max_dim = n_components if n_components is not None else int(all_dims.max())
    local_bases = np.zeros((n, max_dim, d))
    for i in range(n):
        nc = min(all_dims[i], max_dim, all_eigenvectors[i].shape[1])
        local_bases[i, :nc, :] = all_eigenvectors[i][:, :nc].T

    return RobustLocalPCAResult(
        local_bases=local_bases,
        local_eigenvalues=all_eigenvalues,
        local_dims=all_dims,
        local_variances=all_variances,
        outlier_masks=all_outlier_masks,
        condition_numbers=all_condition,
        support_sizes=all_support,
    )
```

- [ ] **Step 5: Run solver-level tests**

Run: `uv run pytest tests/test_robust_local_pca.py::TestRobustLocalPCASolver -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add manylatents/utils/robust_pca_solvers.py tests/test_robust_local_pca.py
git commit -m "feat: add robust local PCA solver (trimmed, MCD, Huber methods)"
```

---

### Task 6: LTSA alignment + PCAModule robust_local integration

**Files:**
- Modify: `manylatents/utils/robust_pca_solvers.py` (add LTSA function)
- Modify: `manylatents/algorithms/latent/pca.py` (implement `_fit_robust_local`)
- Add tests to: `tests/test_robust_local_pca.py`

- [ ] **Step 1: Write PCAModule-level failing tests**

Append to `tests/test_robust_local_pca.py`:

```python
class TestPCAModuleRobustLocal:
    def test_fit_transform_shape(self):
        from manylatents.algorithms.latent.pca import PCAModule
        X, _, _ = make_robust_local_pca_test_data(n=200, contamination_frac=0.0)
        mod = PCAModule(n_components=2, method='robust_local',
                        robust_method='trimmed', n_neighbors=20)
        emb = mod.fit_transform(X)
        assert emb.shape == (200, 2)

    def test_transform_returns_cached(self):
        from manylatents.algorithms.latent.pca import PCAModule
        X, _, _ = make_robust_local_pca_test_data(n=100)
        mod = PCAModule(n_components=2, method='robust_local')
        emb1 = mod.fit_transform(X)
        emb2 = mod.transform(X)
        np.testing.assert_array_equal(emb1, emb2)

    def test_transform_new_data_raises(self):
        from manylatents.algorithms.latent.pca import PCAModule
        X, _, _ = make_robust_local_pca_test_data(n=100)
        mod = PCAModule(n_components=2, method='robust_local')
        mod.fit_transform(X)
        X_new = np.random.default_rng(99).standard_normal((50, 3)).astype(np.float32)
        with pytest.raises(NotImplementedError):
            mod.transform(X_new)

    def test_extra_outputs(self):
        from manylatents.algorithms.latent.pca import PCAModule
        X, _, _ = make_robust_local_pca_test_data(n=100)
        mod = PCAModule(n_components=2, method='robust_local', n_neighbors=15)
        mod.fit_transform(X)
        extras = mod.extra_outputs()
        assert 'local_eigenvalues' in extras
        assert 'local_dims' in extras
        assert 'outlier_masks' in extras
        assert extras['local_dims'].shape == (100,)

    def test_kernel_matrix_raises(self):
        from manylatents.algorithms.latent.pca import PCAModule
        X, _, _ = make_robust_local_pca_test_data(n=100)
        mod = PCAModule(n_components=2, method='robust_local')
        mod.fit_transform(X)
        with pytest.raises(NotImplementedError):
            mod.kernel_matrix()

    def test_api_integration(self):
        from manylatents.api import run
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(300, random_state=42)
        result = run(
            input_data=X.astype(np.float32),
            algorithms={'latent': {
                '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                'n_components': 2,
                'method': 'robust_local',
                'robust_method': 'trimmed',
                'n_neighbors': 20,
            }}
        )
        assert result['embeddings'].shape == (300, 2)
```

- [ ] **Step 2: Implement LTSA alignment**

Add to `manylatents/utils/robust_pca_solvers.py`:

```python
def ltsa_align(X: np.ndarray, indices: np.ndarray,
               local_bases: np.ndarray, n_components: int) -> np.ndarray:
    """Local Tangent Space Alignment (Zhang & Zha 2004).

    Args:
        X: (n, d) data matrix.
        indices: (n, k) neighbor index array.
        local_bases: (n, n_components, d) local PC bases per point.
        n_components: Embedding dimensionality.

    Returns:
        (n, n_components) embedding.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    n, k = indices.shape

    # COO construction (much faster than lil_matrix element access)
    rows, cols, vals = [], [], []

    for i in range(n):
        I_i = indices[i]
        X_local = X[I_i] - X[I_i].mean(axis=0)
        U_i = local_bases[i].T  # (d, n_components)
        Theta_i = X_local @ U_i  # (k, n_components)

        G_i = np.hstack([np.ones((k, 1)), Theta_i])  # (k, n_components + 1)
        W_i = np.eye(k) - G_i @ np.linalg.pinv(G_i)  # (k, k)

        # Vectorized index construction for this neighborhood
        row_idx = np.repeat(I_i, k)
        col_idx = np.tile(I_i, k)
        rows.append(row_idx)
        cols.append(col_idx)
        vals.append(W_i.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    B_csr = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    eigenvalues, eigenvectors = eigsh(B_csr, k=n_components + 1, which='SM')

    # Sort ascending, skip trivial zero eigenvector (index 0)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    embedding = eigenvectors[:, 1:n_components + 1]

    return embedding
```

- [ ] **Step 3: Implement _fit_robust_local in PCAModule**

Replace the `_fit_robust_local` placeholder in `manylatents/algorithms/latent/pca.py`:

```python
def _fit_robust_local(self, x_np):
    from manylatents.utils.robust_pca_solvers import robust_local_pca, ltsa_align
    from manylatents.utils.knn import compute_knn

    self._X_fit_shape = tuple(x_np.shape)

    # Compute neighborhoods
    dists, indices = compute_knn(x_np.astype(np.float32),
                                  k=self.n_neighbors, include_self=False)

    # Robust local PCA
    local_result = robust_local_pca(
        x_np, n_neighbors=self.n_neighbors,
        n_components=self.n_components,
        robust_method=self.robust_method,
        support_fraction=self.support_fraction,
        trim_fraction=self.trim_fraction,
        precomputed_neighbors=indices,
        precomputed_distances=dists,
        random_state=self.init_seed,
    )
    self._local_result = local_result

    # LTSA alignment
    self._embedding = ltsa_align(
        x_np.astype(np.float64), indices,
        local_result.local_bases, self.n_components
    ).astype(x_np.dtype)
```

Also update `extra_outputs()` in `PCAModule` to handle `robust_local`:

```python
# Add this block inside extra_outputs(), after the global RPCA block:
if self.method == 'robust_local' and hasattr(self, '_local_result'):
    r = self._local_result
    extras['local_eigenvalues'] = r.local_eigenvalues
    extras['local_dims'] = r.local_dims
    extras['local_variances'] = r.local_variances
    extras['outlier_masks'] = r.outlier_masks
    extras['condition_numbers'] = r.condition_numbers
    extras['support_sizes'] = r.support_sizes
```

- [ ] **Step 4: Run all robust local PCA tests**

Run: `uv run pytest tests/test_robust_local_pca.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add manylatents/utils/robust_pca_solvers.py manylatents/algorithms/latent/pca.py tests/test_robust_local_pca.py
git commit -m "feat: add robust local PCA + LTSA alignment to PCAModule"
```

---

### Task 7: Hydra config for robust local PCA

**Files:**
- Create: `manylatents/configs/algorithms/latent/robust_local_pca.yaml`

- [ ] **Step 1: Create the config**

```yaml
_target_: manylatents.algorithms.latent.pca.PCAModule
n_components: 2
random_state: ${seed}
neighborhood_size: ${neighborhood_size}
method: robust_local
n_neighbors: 20
robust_method: trimmed
support_fraction: 0.75
trim_fraction: 0.1
verbose: false
```

- [ ] **Step 2: Verify auto-discovery works**

Run: `uv run pytest tests/test_module_instantiation.py -v -k robust_local`
Expected: The new YAML is auto-discovered and PCAModule instantiates

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add manylatents/configs/algorithms/latent/robust_local_pca.yaml
git commit -m "feat: add robust_local_pca.yaml Hydra config"
```

---

### Task 8: Final integration verification

**Files:** None new — run the full suite and verify everything works end-to-end.

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -x -q && uv run pytest manylatents/callbacks/tests/ -x -q
```
Expected: All pass

- [ ] **Step 2: Smoke test CLI**

```bash
uv run python -m manylatents.main algorithms/latent=robust_pca data=swissroll
uv run python -m manylatents.main algorithms/latent=robust_local_pca data=swissroll
```
Expected: Both run without error

- [ ] **Step 3: Smoke test API (notebook-style)**

```bash
uv run python -c "
from manylatents.api import run
import numpy as np
X = np.random.default_rng(42).standard_normal((200, 50)).astype(np.float32)
r = run(input_data=X, algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 5, 'method': 'robust_ialm'}})
print('Global RPCA:', r['embeddings'].shape)
from sklearn.datasets import make_swiss_roll
X2, _ = make_swiss_roll(300, random_state=42)
r2 = run(input_data=X2.astype(np.float32), algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 2, 'method': 'robust_local', 'robust_method': 'trimmed', 'n_neighbors': 20}})
print('Local RPCA:', r2['embeddings'].shape)
"
```
Expected: `Global RPCA: (200, 5)` and `Local RPCA: (300, 2)`
