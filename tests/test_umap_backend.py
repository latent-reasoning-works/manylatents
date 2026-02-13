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
