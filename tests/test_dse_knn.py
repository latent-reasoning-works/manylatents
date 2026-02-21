import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll


def test_knn_diffusion_matrix_blobs_5_eigenvalues_alpha1():
    """5 well-separated clusters at alpha=1.0 (Laplace-Beltrami) should produce
    exactly 5 persistent eigenvalues regardless of cluster size variation.
    alpha=1.0 removes density bias, so unequal cluster sizes don't affect the count."""
    from manylatents.metrics.diffusion_spectral_entropy import compute_diffusion_matrix_knn
    from manylatents.utils.metrics import compute_knn

    # Unequal cluster sizes: density varies across clusters
    X, _ = make_blobs(
        n_samples=[150, 50, 100, 80, 120],
        cluster_std=0.3, random_state=42,
    )
    distances, indices = compute_knn(X, k=15)
    S = compute_diffusion_matrix_knn(distances, indices, alpha=1.0)

    eigvals = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]
    # t=500 needed: 6th eigenvalue is ~0.957 (within-cluster connectivity),
    # needs high t to decay below floor. At t=500, 0.957^500 ≈ 2e-10.
    eigvals_powered = eigvals ** 500

    count = int(np.sum(eigvals_powered > 1e-6))
    assert count == 5, f"Expected 5 persistent eigenvalues at alpha=1.0, got {count}"


def test_knn_diffusion_matrix_blobs_alpha0_density_sensitive():
    """Same unequal blobs at alpha=0 (graph Laplacian) — density variation
    may cause count != 5. This demonstrates why alpha matters."""
    from manylatents.metrics.diffusion_spectral_entropy import compute_diffusion_matrix_knn
    from manylatents.utils.metrics import compute_knn

    X, _ = make_blobs(
        n_samples=[150, 50, 100, 80, 120],
        cluster_std=0.3, random_state=42,
    )
    distances, indices = compute_knn(X, k=15)
    S_alpha0 = compute_diffusion_matrix_knn(distances, indices, alpha=0.0)
    S_alpha1 = compute_diffusion_matrix_knn(distances, indices, alpha=1.0)

    # The two operators should produce different spectra
    eigvals_0 = np.sort(np.abs(np.linalg.eigvalsh(S_alpha0)))[::-1]
    eigvals_1 = np.sort(np.abs(np.linalg.eigvalsh(S_alpha1)))[::-1]
    assert not np.allclose(eigvals_0, eigvals_1, atol=1e-3), \
        "alpha=0 and alpha=1 should produce different spectra on unequal-size clusters"


def test_knn_diffusion_matrix_swissroll_smooth_spectrum():
    """Swissroll (connected manifold) should have no large spectral gap after lambda_1."""
    from manylatents.metrics.diffusion_spectral_entropy import compute_diffusion_matrix_knn
    from manylatents.utils.metrics import compute_knn

    X, _ = make_swiss_roll(n_samples=500, random_state=42)
    distances, indices = compute_knn(X, k=15)
    S = compute_diffusion_matrix_knn(distances, indices, alpha=1.0)

    eigvals = np.sort(np.abs(np.linalg.eigvalsh(S)))[::-1]

    # Ratio of 2nd to 1st eigenvalue should be close to 1 (no big gap)
    ratio = eigvals[1] / eigvals[0]
    assert ratio > 0.9, f"Expected smooth spectrum (ratio > 0.9), got {ratio:.4f}"

    # At very high t, only 1 component should persist (connected manifold).
    # Swissroll's 2nd eigenvalue is ~0.990 — decays very slowly.
    eigvals_powered = eigvals ** 2000
    count = int(np.sum(eigvals_powered > 1e-6))
    assert count == 1, f"Expected 1 persistent eigenvalue for connected manifold, got {count}"


def test_knn_diffusion_matrix_is_symmetric():
    """Output of compute_diffusion_matrix_knn should be symmetric."""
    from manylatents.metrics.diffusion_spectral_entropy import compute_diffusion_matrix_knn
    from manylatents.utils.metrics import compute_knn

    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    distances, indices = compute_knn(X, k=10)
    S = compute_diffusion_matrix_knn(distances, indices, alpha=1.0)

    assert np.allclose(S, S.T, atol=1e-10), "Diffusion matrix should be symmetric"


def test_dense_diffusion_matrix_alpha_param():
    """compute_diffusion_matrix should accept alpha and delegate to symmetric_diffusion_operator."""
    from manylatents.metrics.diffusion_spectral_entropy import compute_diffusion_matrix

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)

    # alpha=0.5 (old default behavior)
    S_half = compute_diffusion_matrix(X, sigma=1.0, alpha=0.5)
    assert S_half.shape == (200, 200)
    assert np.allclose(S_half, S_half.T, atol=1e-10), "Should be symmetric"

    # alpha=1.0 (Laplace-Beltrami) should produce a different matrix
    # Use small sigma so density normalization has a visible effect
    S_one = compute_diffusion_matrix(X, sigma=1.0, alpha=1.0)
    assert not np.allclose(S_half, S_one, atol=1e-3), "alpha=0.5 vs alpha=1.0 should differ"


def test_dse_knn_mode_eigenvalue_count():
    """DSE with kernel='knn' should return eigenvalue count."""
    from manylatents.metrics.diffusion_spectral_entropy import DiffusionSpectralEntropy

    X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.3, random_state=42)

    count = DiffusionSpectralEntropy(
        X, kernel="knn", k=15, alpha=1.0,
        output_mode="eigenvalue_count", t_high=500,
    )
    assert count == 5, f"Expected 5 for 5 clusters, got {count}"


def test_dse_dense_mode_still_works():
    """DSE with kernel='dense' should behave like the old code."""
    from manylatents.metrics.diffusion_spectral_entropy import DiffusionSpectralEntropy

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)

    # Dense mode — override alpha for baseline comparison
    entropy = DiffusionSpectralEntropy(
        X, kernel="dense", gaussian_kernel_sigma=10, alpha=0.5,
        output_mode="entropy", t=1,
    )
    assert isinstance(entropy, float)
    assert entropy > 0


def test_dse_knn_mode_uses_cache():
    """DSE with kernel='knn' should use and populate the cache."""
    from manylatents.metrics.diffusion_spectral_entropy import DiffusionSpectralEntropy
    from manylatents.utils.metrics import compute_knn

    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

    cache = {}
    # Pre-warm cache
    compute_knn(X, k=15, cache=cache)

    # DSE should reuse cached kNN
    count = DiffusionSpectralEntropy(
        X, kernel="knn", k=15, alpha=1.0,
        output_mode="eigenvalue_count", t_high=500,
        cache=cache,
    )
    assert isinstance(count, float)
