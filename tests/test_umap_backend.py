"""Tests for UMAPModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_umap_default_backend_unchanged():
    """UMAPModule with default backend uses umap-learn."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, n_neighbors=5, n_epochs=10)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    emb = m.transform(x)
    assert emb.shape == (50, 2)
    # Should have umap-learn model
    from umap import UMAP as UmapLearnUMAP
    assert isinstance(m.model, UmapLearnUMAP)


def test_umap_accepts_backend_param():
    """UMAPModule accepts backend/device without error."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_umap_torchdr_backend_fit_transform():
    """UMAPModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(
        n_components=2, random_state=42, n_neighbors=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_umap_torchdr_affinity_tensor():
    """UMAPModule with torchdr backend exposes affinity_tensor()."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(
        n_components=2, random_state=42, n_neighbors=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    m.fit_transform(x)
    t = m.affinity_tensor()
    assert isinstance(t, torch.Tensor)


def test_umap_torchdr_not_installed_raises():
    """UMAPModule with torchdr backend raises if not installed."""
    import manylatents.utils.backend as backend_mod
    from manylatents.algorithms.latent.umap import UMAPModule

    # Temporarily fake torchdr as unavailable
    original = backend_mod._torchdr_available
    backend_mod._torchdr_available = False

    try:
        with pytest.raises(ImportError, match="torchdr"):
            UMAPModule(
                n_components=2, random_state=42,
                backend="torchdr", device="cpu",
            )
    finally:
        backend_mod._torchdr_available = original


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_umap_backend_output_agreement():
    """CPU and TorchDR UMAP produce structurally consistent affinity matrices."""
    from manylatents.algorithms.latent.umap import UMAPModule

    x = torch.randn(80, 15, generator=torch.Generator().manual_seed(42))

    m_cpu = UMAPModule(n_components=2, random_state=42, n_neighbors=10, backend=None)
    emb_cpu = m_cpu.fit_transform(x)
    A_cpu = m_cpu.affinity_matrix(use_symmetric=True)
    K_cpu = m_cpu.kernel_matrix()

    m_tdr = UMAPModule(n_components=2, random_state=42, n_neighbors=10, backend="torchdr", device="cpu")
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


def test_umap_negative_sample_rate_affects_embedding():
    """negative_sample_rate should reach sklearn UMAP and change the embedding."""
    from manylatents.algorithms.latent.umap import UMAPModule

    x = torch.randn(80, 10, generator=torch.Generator().manual_seed(0))

    m1 = UMAPModule(
        n_components=2, random_state=42, n_neighbors=10,
        n_epochs=50, negative_sample_rate=1,
    )
    emb1 = m1.fit_transform(x)

    m2 = UMAPModule(
        n_components=2, random_state=42, n_neighbors=10,
        n_epochs=50, negative_sample_rate=10,
    )
    emb2 = m2.fit_transform(x)

    # Same seed, same data, different negative_sample_rate -> different embeddings
    assert emb1.shape == emb2.shape == (80, 2)
    assert not np.allclose(emb1, emb2, atol=1e-3), (
        "Embeddings should differ when negative_sample_rate changes"
    )


def test_umap_negative_sample_rate_none_uses_default():
    """negative_sample_rate=None should not change default sklearn behavior."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, n_neighbors=5, n_epochs=10)
    assert m.negative_sample_rate is None
    # sklearn default is 5 — verify it's not overridden
    from umap import UMAP as SklearnUMAP
    assert m.model.negative_sample_rate == SklearnUMAP().negative_sample_rate
