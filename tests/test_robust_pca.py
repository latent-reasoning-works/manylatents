"""Tests for global Robust PCA solvers and PCAModule integration."""
import numpy as np
import pytest


def make_rpca_test_data(m=200, n=100, rank=5, sparse_frac=0.05,
                         noise_std=0.0, seed=42):
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


class TestPCAModuleRobust:
    def test_standard_pca_unchanged(self):
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
        assert 'kernel' in extras

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
        extras = mod.extra_outputs()
        assert extras['low_rank_matrix'].shape == (100, 100)
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
