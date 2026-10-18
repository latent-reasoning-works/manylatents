"""Unified input_metric: one key routes the input distance to UMAP, t-SNE, PHATE and Isomap."""
from pathlib import Path

import hydra
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from scipy.spatial.distance import cdist

from manylatents.algorithms.latent.isomap import IsomapModule
from manylatents.algorithms.latent.pca import PCAModule
from manylatents.algorithms.latent.phate import PHATEModule
from manylatents.algorithms.latent.tsne import TSNEModule
from manylatents.algorithms.latent.umap import UMAPModule
import manylatents.configs  # noqa: F401  (registers base_config with the ConfigStore)
from manylatents.api import run

CONFIGS = Path(__file__).resolve().parents[1] / "manylatents" / "configs"


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(loc=c, size=(150, 5)) for c in (0.0, 6.0)])


def _recall(y, x, q=10):
    a = np.argsort(cdist(y, y), 1)[:, 1:q + 1]
    b = np.argsort(cdist(x, x), 1)[:, 1:q + 1]
    return float(np.mean([len(set(a[i]) & set(b[i])) / q for i in range(len(x))]))


def test_default_keeps_each_module_metric():
    assert UMAPModule().metric == "euclidean"
    assert TSNEModule().metric == "euclidean"
    assert PHATEModule().knn_dist == "euclidean"
    assert IsomapModule().model.metric == "minkowski"


def test_input_metric_reaches_every_backend():
    assert UMAPModule(input_metric="precomputed").metric == "precomputed"
    assert TSNEModule(input_metric="precomputed").metric == "precomputed"
    m = PHATEModule(input_metric="precomputed", n_landmark=None)
    assert m.knn_dist == "precomputed_distance" and m.model.n_pca is None
    assert IsomapModule(input_metric="precomputed").model.metric == "precomputed"
    assert UMAPModule(input_metric="cosine").metric == "cosine"


def test_unsupported_module_raises_instead_of_ignoring():
    PCAModule(input_metric="euclidean")
    with pytest.raises(ValueError, match="input_metric"):
        PCAModule(input_metric="precomputed")


def test_tsne_pca_start_on_distance_matrix_raises():
    with pytest.raises(ValueError, match="initialization"):
        TSNEModule(input_metric="precomputed", initialization="pca")


@pytest.mark.parametrize("name,kw,tol", [
    ("umap", {"n_epochs": 100}, 0.08),
    ("tsne", {"n_iter_late": 300}, 0.08),
    ("phate", {"n_landmark": None, "n_pca": None, "t": 10}, 0.05),
    ("isomap", {}, 0.02),
])
def test_precomputed_euclidean_matches_features_through_run(blobs, name, kw, tol):
    """A precomputed Euclidean matrix keeps about as many neighbours as the features route."""
    x = blobs.astype(np.float32)
    from manylatents.algorithms.latent import get_algorithm
    tgt = f"{get_algorithm(name).__module__}.{get_algorithm(name).__name__}"
    cfg = lambda **extra: {"latent": {"_target_": tgt, "random_state": 0, "neighborhood_size": 15, **kw, **extra}}
    feat = np.asarray(run(input_data=x, algorithms=cfg(), seed=0)["embeddings"])
    pre = np.asarray(run(input_data=cdist(x, x).astype(np.float32), algorithms=cfg(input_metric="precomputed"),
                         seed=0)["embeddings"])
    assert pre.shape == feat.shape
    assert abs(_recall(pre, x) - _recall(feat, x)) <= tol


@pytest.mark.parametrize("algo,attr", [("umap", "metric"), ("tsne", "metric"), ("phate", "knn_dist")])
def test_hydra_key_sweeps_all_methods(algo, attr):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        extra = ["algorithms.latent.initialization=random"] if algo == "tsne" else []   # the config's PCA start cannot take a distance matrix
        cfg = compose(config_name="config", overrides=[f"algorithms/latent={algo}", "input_metric=precomputed", *extra])
    mod = hydra.utils.instantiate(cfg.algorithms.latent)
    assert getattr(mod, attr) in ("precomputed", "precomputed_distance")
