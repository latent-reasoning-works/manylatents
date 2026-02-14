"""Tests for TSNEModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_tsne_default_backend_unchanged():
    """TSNEModule with default backend uses openTSNE."""
    from manylatents.algorithms.latent.tsne import TSNEModule

    m = TSNEModule(n_components=2, random_state=42, perplexity=10, n_iter_early=50, n_iter_late=50)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    # Should have openTSNE embedding_train
    assert hasattr(m, 'embedding_train')


def test_tsne_accepts_backend_param():
    """TSNEModule accepts backend/device without error."""
    from manylatents.algorithms.latent.tsne import TSNEModule

    m = TSNEModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_tsne_torchdr_backend_fit_transform():
    """TSNEModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.tsne import TSNEModule

    m = TSNEModule(
        n_components=2, random_state=42, perplexity=10,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


def test_tsne_torchdr_not_installed_raises():
    """TSNEModule with torchdr backend raises if not installed."""
    import manylatents.utils.backend as backend_mod
    from manylatents.algorithms.latent.tsne import TSNEModule

    original = backend_mod._torchdr_available
    backend_mod._torchdr_available = False

    try:
        with pytest.raises(ImportError, match="torchdr"):
            TSNEModule(
                n_components=2, random_state=42,
                backend="torchdr", device="cpu",
            )
    finally:
        backend_mod._torchdr_available = original


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_tsne_backend_output_agreement():
    """CPU and TorchDR TSNE produce structurally consistent affinity matrices."""
    from manylatents.algorithms.latent.tsne import TSNEModule

    x = torch.randn(80, 15, generator=torch.Generator().manual_seed(42))

    m_cpu = TSNEModule(n_components=2, random_state=42, perplexity=10, backend=None)
    emb_cpu = m_cpu.fit_transform(x)
    A_cpu = m_cpu.affinity_matrix(use_symmetric=True)
    K_cpu = m_cpu.kernel_matrix()

    m_tdr = TSNEModule(n_components=2, random_state=42, perplexity=10, backend="torchdr", device="cpu")
    emb_tdr = m_tdr.fit_transform(x)
    A_tdr = m_tdr.affinity_matrix(use_symmetric=True)
    K_tdr = m_tdr.kernel_matrix()

    # Shape
    assert emb_cpu.shape == emb_tdr.shape == (80, 2)
    assert A_cpu.shape == A_tdr.shape == (80, 80)

    # Symmetry
    assert np.allclose(A_cpu, A_cpu.T, atol=1e-6), "CPU affinity not symmetric"
    assert np.allclose(A_tdr, A_tdr.T, atol=1e-6), "TorchDR affinity not symmetric"

    # Eigenvalue spectrum bounded
    eigs_cpu = np.sort(np.linalg.eigvalsh(A_cpu))[::-1]
    eigs_tdr = np.sort(np.linalg.eigvalsh(A_tdr))[::-1]
    assert eigs_cpu.max() <= 1.01, f"CPU eigenvalue above 1: {eigs_cpu.max()}"
    assert eigs_tdr.max() <= 1.01, f"TorchDR eigenvalue above 1: {eigs_tdr.max()}"
    assert eigs_cpu.min() >= -1.01, f"CPU eigenvalue below -1: {eigs_cpu.min()}"
    assert eigs_tdr.min() >= -1.01, f"TorchDR eigenvalue below -1: {eigs_tdr.min()}"

    # Kernel non-negative
    assert K_cpu.min() >= -1e-7, f"CPU kernel has negative values: {K_cpu.min()}"
    assert K_tdr.min() >= -1e-7, f"TorchDR kernel has negative values: {K_tdr.min()}"
