"""Tests for SammonModule (Sammon mapping)."""
import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.sammon import SammonModule


@pytest.fixture
def data():
    return torch.randn(100, 10, generator=torch.Generator().manual_seed(0))


def test_isinstance_latent_module():
    m = SammonModule(n_components=2, random_state=42)
    assert isinstance(m, LatentModule)


def test_fit_transform(data):
    m = SammonModule(n_components=2, random_state=42)
    emb = m.fit_transform(data)
    assert emb.shape == (100, 2)


def test_determinism(data):
    m1 = SammonModule(n_components=2, random_state=42)
    m2 = SammonModule(n_components=2, random_state=42)
    e1 = m1.fit_transform(data)
    e2 = m2.fit_transform(data)
    assert np.allclose(e1.numpy(), e2.numpy())


def test_fit_then_transform(data):
    m = SammonModule(n_components=2, random_state=42)
    m.fit(data)
    emb = m.transform(data)
    assert emb.shape == (100, 2)


def test_kernel_matrix(data):
    m = SammonModule(n_components=2, random_state=42)
    m.fit_transform(data)
    K = m.kernel()
    assert K.shape == (100, 100)
    # Gram matrix should be symmetric.
    assert np.allclose(K, K.T)
