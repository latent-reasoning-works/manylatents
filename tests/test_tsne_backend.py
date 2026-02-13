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
