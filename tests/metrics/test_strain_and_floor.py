"""Tests for the per-point local strain and covariance-floor metrics.

Both are per-point reads of a point's k-NN neighborhood:
  LocalStrain     -- localized classical-MDS strain of an embedding against the high-dim input
  CovarianceFloor -- Phi(d), the Eckart-Young discarded tail mass of the local Gram spectrum

The tangent participation ratio is NOT defined here: LocalSpectralAnalysis already provides it
(aliases "participation_ratio" / "pr" / "effective_rank") and is bit-identical for this purpose.
"""
import numpy as np
import pytest

from manylatents.metrics.covariance_floor import CovarianceFloor
from manylatents.metrics.strain import LocalStrain


def _dataset(X):
    """The metrics take the high-dimensional input via an object exposing .data."""
    return type("D", (), {"data": X})()


@pytest.fixture
def plane_in_50d():
    """An exactly 2-D configuration embedded isometrically in R^50.

    Returns (P, X): P is the true 2-D coordinate set, X its image under an orthonormal map, so a
    faithful 2-D embedding of X is P itself (up to the similarity the strain quotients out).
    """
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 2))
    Q, _ = np.linalg.qr(rng.standard_normal((50, 50)))
    X = P @ Q[:, :2].T
    return P, X


def test_strain_is_zero_for_an_isometric_embedding(plane_in_50d):
    """Recovering the plane's own coordinates is a perfect local embedding."""
    P, X = plane_in_50d
    strain = LocalStrain(P, dataset=_dataset(X), n_neighbors=20, return_per_sample=True)
    assert strain.shape == (len(X),)
    assert np.max(strain) < 1e-9


def test_strain_is_scale_invariant(plane_in_50d):
    """The optimal scalar removes the embedding's arbitrary global scale."""
    _, X = plane_in_50d
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((len(X), 2))
    a = LocalStrain(Y, dataset=_dataset(X), n_neighbors=20, return_per_sample=True)
    b = LocalStrain(Y * 37.0, dataset=_dataset(X), n_neighbors=20, return_per_sample=True)
    assert np.allclose(a, b, atol=1e-12)


def test_strain_mean_matches_per_sample_mean(plane_in_50d):
    _, X = plane_in_50d
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((len(X), 2))
    per = LocalStrain(Y, dataset=_dataset(X), n_neighbors=20, return_per_sample=True)
    assert float(LocalStrain(Y, dataset=_dataset(X), n_neighbors=20)) == pytest.approx(per.mean())


def test_floor_is_zero_on_an_exactly_two_dimensional_neighborhood(plane_in_50d):
    _, X = plane_in_50d
    phi = CovarianceFloor(X, n_neighbors=20, target_dim=2, return_per_sample=True)
    assert phi.shape == (len(X),)
    assert np.max(phi) < 1e-12


def test_floor_is_in_unit_interval_and_decreases_with_target_dim():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 10)) @ rng.standard_normal((10, 40))
    phi2 = CovarianceFloor(X, n_neighbors=30, target_dim=2, return_per_sample=True)
    phi3 = CovarianceFloor(X, n_neighbors=30, target_dim=3, return_per_sample=True)
    assert phi2.min() >= 0.0 and phi2.max() <= 1.0
    assert np.all(phi3 <= phi2 + 1e-12)


def test_floor_squares_the_eigenvalues():
    """Phi(d) uses lambda^2 (Frobenius error of the Gram), not the variance ratio."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 8)) @ rng.standard_normal((8, 30))
    k = 25
    phi = CovarianceFloor(X, n_neighbors=k, target_dim=2, return_per_sample=True)

    from manylatents.utils.knn import compute_knn

    _, idx = compute_knn(np.ascontiguousarray(X, dtype=np.float32), k=k, include_self=False)
    expected = np.empty(len(X))
    for i in range(len(X)):
        nb = X[idx[i]]
        s = np.linalg.svd(nb - nb.mean(0), compute_uv=False)
        lam2 = (s.astype(np.float64) ** 2) ** 2
        expected[i] = lam2[2:].sum() / lam2.sum()
    assert np.allclose(phi, expected, rtol=1e-12, atol=1e-14)


def test_include_self_defaults_are_the_documented_ones():
    """Strain keeps the point in its neighborhood; the covariance floor does not."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((150, 5)) @ rng.standard_normal((5, 20))
    Y = rng.standard_normal((150, 2))
    k = 20

    s_default = LocalStrain(Y, dataset=_dataset(X), n_neighbors=k, return_per_sample=True)
    s_self = LocalStrain(Y, dataset=_dataset(X), n_neighbors=k, include_self=True, return_per_sample=True)
    assert np.allclose(s_default, s_self)

    phi_default = CovarianceFloor(X, n_neighbors=k, return_per_sample=True)
    phi_noself = CovarianceFloor(X, n_neighbors=k, include_self=False, return_per_sample=True)
    assert np.allclose(phi_default, phi_noself)

