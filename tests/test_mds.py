"""Tests for MDSModule (Multidimensional Scaling)."""
import numpy as np
import pytest
import torch

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.multi_dimensional_scaling import MDSModule


@pytest.fixture
def data():
    return torch.randn(100, 10, generator=torch.Generator().manual_seed(0))


def test_isinstance_latent_module():
    m = MDSModule(n_components=2, random_state=42)
    assert isinstance(m, LatentModule)


def test_fit_transform_classic(data):
    m = MDSModule(n_components=2, random_state=42, how="classic")
    emb = m.fit_transform(data)
    assert emb.shape == (100, 2)


def test_fit_transform_metric_smacof(data):
    m = MDSModule(n_components=2, random_state=42, how="metric", solver="smacof")
    emb = m.fit_transform(data)
    assert emb.shape == (100, 2)


def test_fit_transform_metric_sgd(data):
    pytest.importorskip("phate")
    m = MDSModule(n_components=2, random_state=42, how="metric", solver="sgd")
    emb = m.fit_transform(data)
    assert emb.shape == (100, 2)


def test_determinism(data):
    m1 = MDSModule(n_components=2, random_state=42, how="classic")
    m2 = MDSModule(n_components=2, random_state=42, how="classic")
    e1 = m1.fit_transform(data)
    e2 = m2.fit_transform(data)
    np.testing.assert_array_equal(e1.numpy(), e2.numpy())


def test_kernel_matrix(data):
    m = MDSModule(n_components=2, random_state=42, how="classic")
    m.fit_transform(data)
    K = m.kernel_matrix()
    assert K.shape == (100, 100)


def test_fit_then_transform(data):
    m = MDSModule(n_components=2, random_state=42, how="classic")
    m.fit(data)
    emb = m.transform(data)
    assert emb.shape == (100, 2)
