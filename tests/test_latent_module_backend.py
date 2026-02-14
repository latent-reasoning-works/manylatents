"""Tests for LatentModule backend/device parameters."""
import numpy as np
import pytest
import torch


def test_latent_module_accepts_backend_param():
    """LatentModule.__init__ accepts backend parameter."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x

    m = TestModule(n_components=2, backend="torchdr", device="cpu")
    assert m.backend == "torchdr"
    assert m.device == "cpu"


def test_latent_module_defaults_none():
    """Backend and device default to None."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x

    m = TestModule()
    assert m.backend is None
    assert m.device is None


def test_affinity_tensor_from_numpy():
    """affinity_tensor() converts numpy affinity to torch.Tensor."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x
        def affinity_matrix(self, ignore_diagonal=False, use_symmetric=False):
            return np.eye(5, dtype=np.float64)

    m = TestModule()
    t = m.affinity_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.shape == (5, 5)


def test_existing_modules_unaffected():
    """Existing modules still work without backend/device."""
    from manylatents.algorithms.latent.umap import UMAPModule

    # Should not raise — backend/device go to **kwargs -> base class
    m = UMAPModule(n_components=2, random_state=42)
    assert m.backend is None
    assert m.device is None
