"""Test that TorchDR DiffusionMap backend produces embeddings consistent with graphtools."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_swiss_roll


@pytest.fixture
def swiss_roll_data():
    X, _ = make_swiss_roll(n_samples=500, random_state=42)
    return X.astype(np.float32)


def test_torchdr_eigenvalues_match_graphtools(swiss_roll_data):
    """Top eigenvalues from TorchDR backend should be close to graphtools backend."""
    pytest.importorskip("torchdr")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule

    x = torch.from_numpy(swiss_roll_data)

    dm_gt = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                                n_landmark=None)
    dm_gt.fit(x)
    evals_gt = dm_gt.model.evals[:10]

    dm_td = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                                backend="torchdr", device="cpu", n_landmark=None)
    dm_td.fit(x)
    evals_td = dm_td._torchdr_evals[:10]

    # Top eigenvalues should agree within 10% (different kernels, similar spectrum)
    # Both should have lambda_1 ≈ 1.0 and a gradual decay
    assert np.isclose(evals_gt[0], 1.0, atol=0.01), f"graphtools lambda_1={evals_gt[0]}"
    assert np.isclose(evals_td[0], 1.0, atol=0.01), f"torchdr lambda_1={evals_td[0]}"

    # Spectral shape correlation: eigenvalue decay profiles should be similar
    corr = np.corrcoef(evals_gt[:10], evals_td[:10])[0, 1]
    assert corr > 0.9, f"Eigenvalue profile correlation={corr:.3f}, expected > 0.9"


def test_torchdr_embedding_not_degenerate(swiss_roll_data):
    """TorchDR embedding should have meaningful spread, not collapse to zero."""
    pytest.importorskip("torchdr")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule

    dm = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                             backend="torchdr", device="cpu", n_landmark=None)
    x = torch.from_numpy(swiss_roll_data)
    dm.fit(x)
    emb = dm.transform(x).detach().cpu().numpy()

    # Embedding should have nonzero spread in both dimensions
    assert emb[:, 0].std() > 1e-6, "Embedding dim 0 collapsed"
    assert emb[:, 1].std() > 1e-6, "Embedding dim 1 collapsed"


def test_torchdr_embedding_correlates_with_graphtools(swiss_roll_data):
    """Embeddings from both backends should capture similar structure."""
    pytest.importorskip("torchdr")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule

    x = torch.from_numpy(swiss_roll_data)

    dm_gt = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                                n_landmark=None)
    dm_gt.fit(x)
    emb_gt = dm_gt.transform(x).detach().cpu().numpy()

    dm_td = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                                backend="torchdr", device="cpu", n_landmark=None)
    dm_td.fit(x)
    emb_td = dm_td.transform(x).detach().cpu().numpy()

    # Pairwise distance matrices should correlate (structure preservation).
    # Threshold is 0.3 (not 0.9) because the kernels differ:
    # graphtools uses alpha-decay with graphtools-specific kNN construction,
    # TorchDR uses a simplified alpha-decay with cdist-based distances.
    # The key invariant is that both capture manifold structure (non-random),
    # not that they are numerically identical.
    from scipy.spatial.distance import pdist
    d_gt = pdist(emb_gt)
    d_td = pdist(emb_td)
    corr = np.corrcoef(d_gt, d_td)[0, 1]
    assert corr > 0.3, (
        f"Pairwise distance correlation={corr:.3f}, expected > 0.3. "
        f"Embeddings capture unrelated structure."
    )


def test_torchdr_affinity_matrix_works(swiss_roll_data):
    """affinity_matrix() and kernel_matrix() should return valid matrices."""
    pytest.importorskip("torchdr")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule

    dm = DiffusionMapModule(n_components=2, knn=15, t=5, random_state=42,
                             backend="torchdr", device="cpu", n_landmark=None)
    x = torch.from_numpy(swiss_roll_data)
    dm.fit(x)

    n = swiss_roll_data.shape[0]
    A = dm.affinity_matrix(use_symmetric=True)
    assert A.shape == (n, n)
    assert np.allclose(A, A.T, atol=1e-6), "Symmetric affinity not symmetric"

    K = dm.kernel_matrix()
    assert K.shape == (n, n)
    assert np.all(K >= 0), "Kernel has negative entries"
