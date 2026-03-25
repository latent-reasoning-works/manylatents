"""Tests for SelectiveCorrectionModule and EffectiveNeighborhoodSize metric."""

import numpy as np
import torch
import pytest

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.selective_correction import SelectiveCorrectionModule
from manylatents.metrics.effective_neighborhood_size import EffectiveNeighborhoodSize


class DummyModule(LatentModule):
    """Minimal LatentModule for testing — PCA-2D with a synthetic affinity matrix."""

    def __init__(self, **kwargs):
        super().__init__(n_components=2, **kwargs)
        self._affinity = None

    def fit(self, x, y=None):
        x_np = x.detach().cpu().numpy()
        # Simple PCA-2D
        mean = x_np.mean(axis=0)
        centered = x_np - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        self._embedding = centered @ Vt[:2].T
        self._mean = mean
        self._Vt = Vt[:2]

        # Build a simple kNN affinity (k=5, uniform weights)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(x_np)
        dists, indices = nn.kneighbors()
        n = x_np.shape[0]
        W = np.zeros((n, n))
        for i in range(n):
            for j in indices[i]:
                W[i, j] = 1.0 / 5.0
        self._affinity = W
        self._is_fitted = True

    def transform(self, x):
        x_np = x.detach().cpu().numpy()
        emb = (x_np - self._mean) @ self._Vt.T
        return torch.tensor(emb, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal=False, use_symmetric=False):
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        W = self._affinity.copy()
        if ignore_diagonal:
            np.fill_diagonal(W, 0)
        return W


@pytest.fixture
def gaussian_data():
    """Two Gaussian blobs in 10D."""
    rng = np.random.default_rng(42)
    n = 200
    d = 10
    X = np.vstack([
        rng.normal(0, 1, (n // 2, d)),
        rng.normal(3, 1, (n // 2, d)),
    ])
    return torch.tensor(X, dtype=torch.float32)


class TestEffectiveNeighborhoodSize:
    def test_uniform_weights_give_k(self, gaussian_data):
        """With uniform 1/k weights, k_eff should equal k."""
        module = DummyModule()
        module.fit(gaussian_data)
        emb = module.transform(gaussian_data).numpy()

        result = EffectiveNeighborhoodSize(emb, module=module)
        # Uniform weights with k=5 → k_eff ≈ 5 (ignoring diagonal)
        assert "k_eff" in result
        assert "mean_k_eff" in result
        assert result["k_eff"].shape == (len(gaussian_data),)
        # With uniform 1/5 weights on 5 neighbors, PR = (5 * 1/5)^2 / (5 * (1/5)^2) = 1 / (1/5) = 5
        np.testing.assert_allclose(result["k_eff"], 5.0, atol=0.1)

    def test_requires_module(self, gaussian_data):
        """Should raise if no module provided."""
        with pytest.raises(ValueError, match="requires a fitted module"):
            EffectiveNeighborhoodSize(gaussian_data.numpy())


class TestSelectiveCorrectionModule:
    def test_is_latent_module(self):
        inner = DummyModule()
        module = SelectiveCorrectionModule(
            inner=inner, correction_steps=5, diagnostic_k=20,
        )
        assert isinstance(module, LatentModule)

    def test_fit_transform_shape(self, gaussian_data):
        inner = DummyModule()
        module = SelectiveCorrectionModule(
            inner=inner,
            correction_steps=10,
            diagnostic_k=20,
            correction_k=15,
            correction_k_min=3,
            correction_k_steps=5,
        )
        result = module.fit_transform(gaussian_data)
        assert result.shape == (len(gaussian_data), 2)

    def test_extra_outputs_contain_diagnostics(self, gaussian_data):
        inner = DummyModule()
        module = SelectiveCorrectionModule(
            inner=inner,
            correction_steps=10,
            diagnostic_k=20,
            correction_k=15,
            correction_k_min=3,
            correction_k_steps=5,
        )
        module.fit_transform(gaussian_data)
        extras = module.extra_outputs()
        assert "mismatch_labels" in extras
        assert "mismatch_ratio" in extras

    def test_delegates_affinity(self, gaussian_data):
        inner = DummyModule()
        module = SelectiveCorrectionModule(
            inner=inner,
            correction_steps=5,
            diagnostic_k=20,
            correction_k=15,
        )
        module.fit_transform(gaussian_data)
        aff = module.affinity_matrix()
        assert aff.shape == (len(gaussian_data), len(gaussian_data))
