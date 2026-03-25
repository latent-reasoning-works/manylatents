"""Tests for robust local PCA and PCAModule robust_local integration."""
import numpy as np
import pytest


def make_robust_local_pca_test_data(n=500, noise_std=0.0,
                                      contamination_frac=0.05, seed=42):
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
        assert np.median(result.local_dims) == 2
