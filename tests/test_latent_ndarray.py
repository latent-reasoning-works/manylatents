"""Test LatentModule accepts ndarray and returns matching type."""
import numpy as np
import torch
import pytest
from manylatents.algorithms.latent.pca import PCAModule


def test_fit_transform_ndarray_returns_ndarray():
    """ndarray in -> ndarray out."""
    mod = PCAModule(n_components=2)
    X = np.random.randn(50, 10).astype(np.float32)
    result = mod.fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 2)


def test_fit_transform_tensor_returns_tensor():
    """Tensor in -> Tensor out (backward compat)."""
    mod = PCAModule(n_components=2)
    X = torch.randn(50, 10)
    result = mod.fit_transform(X)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (50, 2)


def test_fit_ndarray_transform_ndarray():
    """Separate fit + transform with ndarray."""
    mod = PCAModule(n_components=2)
    X = np.random.randn(50, 10).astype(np.float32)
    mod.fit(X)
    result = mod.transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 2)


def test_fit_ndarray_sets_is_fitted():
    mod = PCAModule(n_components=2)
    X = np.random.randn(50, 10).astype(np.float32)
    mod.fit(X)
    assert mod._is_fitted


def test_umap_fit_transform_ndarray():
    """UMAP overrides fit_transform — verify ndarray works through override."""
    from manylatents.algorithms.latent.umap import UMAPModule
    mod = UMAPModule(n_components=2, n_neighbors=5)
    X = np.random.randn(50, 10).astype(np.float32)
    result = mod.fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 2)
