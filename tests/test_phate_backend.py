"""Tests for PHATEModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_phate_default_backend_unchanged():
    """PHATEModule with default backend uses phate library."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, knn=5, t=5)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    from phate import PHATE
    assert isinstance(m.model, PHATE)


def test_phate_accepts_backend_param():
    """PHATEModule accepts backend/device without error."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_backend_fit_transform():
    """PHATEModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=5, t=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_param_mapping():
    """PHATEModule maps knn->k and decay->alpha for TorchDR."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=10, t=50, decay=40,
        backend="torchdr", device="cpu",
    )
    # TorchDR PHATE uses 'k' not 'knn', 'alpha' not 'decay'
    assert m.model.k == 10
    assert m.model.alpha == 40
    assert m.model.t == 50
