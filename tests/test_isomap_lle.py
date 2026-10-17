"""Tests for IsomapModule and LocallyLinearModule (sklearn backends)."""
import numpy as np
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll

from manylatents.algorithms.latent import get_algorithm
from manylatents.algorithms.latent.isomap import IsomapModule
from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.algorithms.latent.lle import LocallyLinearModule


@pytest.fixture(scope="module")
def roll():
    X, t = make_swiss_roll(n_samples=800, noise=0.05, random_state=0)
    return X.astype(np.float32), t


@pytest.fixture(scope="module")
def small_roll():
    X, t = make_swiss_roll(n_samples=200, noise=0.0, random_state=1)
    return X, t


# ── parameters reach sklearn ──────────────────────────────────────────────────


def test_isomap_params_reach_sklearn():
    m = IsomapModule(n_components=3, n_neighbors=7, path_method="D",
                     eigen_solver="dense", metric="euclidean", p=1)
    params = m.model.get_params()
    assert isinstance(m, LatentModule)
    assert params["n_components"] == 3
    assert params["n_neighbors"] == 7
    assert params["path_method"] == "D"
    assert params["eigen_solver"] == "dense"
    assert params["metric"] == "euclidean"
    assert params["p"] == 1  # routed through **kwargs


def test_lle_params_reach_sklearn():
    m = LocallyLinearModule(n_components=2, method="modified", n_neighbors=12,
                            reg=0.01, random_state=7, eigen_solver="dense")
    params = m.model.get_params()
    assert params["method"] == "modified"
    assert params["n_neighbors"] == 12
    assert params["reg"] == 0.01
    assert params["random_state"] == 7
    assert params["eigen_solver"] == "dense"


@pytest.mark.parametrize("cls", [IsomapModule, LocallyLinearModule])
def test_neighborhood_size_overrides_n_neighbors(cls):
    m = cls(n_neighbors=5, neighborhood_size=23)
    assert m.n_neighbors == 23
    assert m.neighborhood_size == 23
    assert m.model.get_params()["n_neighbors"] == 23


@pytest.mark.parametrize("cls", [IsomapModule, LocallyLinearModule])
def test_unknown_kwarg_raises(cls):
    with pytest.raises(TypeError):
        cls(n_neighbours=5)


def test_lle_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        LocallyLinearModule(method="laplacian")


def test_neighborhood_size_changes_the_fitted_graph(small_roll):
    X, _ = small_roll
    iso = IsomapModule(neighborhood_size=8, eigen_solver="dense")
    iso.fit(X)
    # The fitted kNN graph has exactly k stored neighbors per row.
    assert iso.model.nbrs_.n_neighbors == 8
    assert iso.model.nbrs_.kneighbors(X[:3], return_distance=False).shape == (3, 8)


# ── behaviour on the swiss roll ───────────────────────────────────────────────


def test_isomap_unrolls_swiss_roll(roll):
    X, t = roll
    emb = IsomapModule(n_components=2, neighborhood_size=10).fit_transform(X)
    assert emb.shape == (len(X), 2)
    rho = spearmanr(emb[:, 0], t).statistic
    assert abs(rho) > 0.9, rho


def test_isomap_tensor_in_tensor_out(small_roll):
    X, _ = small_roll
    x = torch.tensor(X, dtype=torch.float32)
    emb = IsomapModule(neighborhood_size=10, eigen_solver="dense").fit_transform(x)
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (len(X), 2)


@pytest.mark.parametrize("method,k", [("standard", 10), ("hessian", 12),
                                      ("modified", 10), ("ltsa", 10)])
def test_lle_methods_run(small_roll, method, k):
    X, _ = small_roll
    m = LocallyLinearModule(method=method, neighborhood_size=k,
                            eigen_solver="dense", random_state=0)
    emb = m.fit_transform(X)
    assert emb.shape == (len(X), 2)
    assert np.all(np.isfinite(emb))
    np.testing.assert_allclose(emb, m.model.embedding_)


def test_lle_transform_new_rows(small_roll):
    X, _ = small_roll
    m = LocallyLinearModule(method="ltsa", neighborhood_size=10, eigen_solver="dense")
    m.fit(X[:150])
    assert m.transform(X[150:]).shape == (50, 2)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        IsomapModule().transform(np.zeros((5, 3)))
    with pytest.raises(RuntimeError):
        LocallyLinearModule().transform(np.zeros((5, 3)))


# ── registry and API ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,cls", [("isomap", IsomapModule),
                                      ("lle", LocallyLinearModule),
                                      ("locally_linear", LocallyLinearModule)])
def test_registry_names(name, cls):
    assert get_algorithm(name) is cls


def test_run_api_isomap(roll):
    from manylatents.api import run

    X, t = roll
    out = run(input_data=X, algorithm="isomap", seed=0, neighborhood_size=10)
    emb = np.asarray(out["embeddings"])
    assert emb.shape == (len(X), 2)
    assert abs(spearmanr(emb[:, 0], t).statistic) > 0.9


def test_run_api_lle_method_kwarg(small_roll):
    from manylatents.api import run

    X, _ = small_roll
    out = run(input_data=X.astype(np.float32), algorithm="lle", method="ltsa",
              seed=0, neighborhood_size=10)
    assert np.asarray(out["embeddings"]).shape == (len(X), 2)
