"""Test DiffusionMap caches L and S from fit."""
import numpy as np
import torch
from manylatents.algorithms.latent.diffusion_map import DiffusionMap, DiffusionMapModule


def _make_data(n=100, d=5, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


def test_diffusion_map_stashes_L_and_S():
    """Inner DiffusionMap.fit() stores L and S."""
    X = _make_data()
    dm = DiffusionMap(n_components=2, knn=5, n_landmark=None)
    dm.fit(X)
    assert hasattr(dm, "L"), "DiffusionMap.fit should stash L (diffusion operator)"
    assert hasattr(dm, "S"), "DiffusionMap.fit should stash S (symmetric operator)"
    assert dm.L.shape == (100, 100)
    assert dm.S.shape == (100, 100)
    row_sums = dm.L.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    np.testing.assert_allclose(dm.S, dm.S.T, atol=1e-10)


def test_diffusionmap_module_affinity_uses_cached():
    """DiffusionMapModule.affinity_matrix() returns cached values, not recomputed."""
    X = _make_data()
    mod = DiffusionMapModule(n_components=2, knn=5, n_landmark=None)
    mod.fit_transform(torch.tensor(X))

    P = mod.affinity_matrix(use_symmetric=False)
    S = mod.affinity_matrix(use_symmetric=True)

    assert P.shape == (100, 100)
    assert S.shape == (100, 100)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    np.testing.assert_allclose(S, mod.model.S, atol=1e-10)
