"""PHATEModule(knn_dist=...) reaches phate, and a precomputed distance matrix gives the same layout as features."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from manylatents.algorithms.latent.phate import PHATEModule


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(loc=c, size=(100, 5)) for c in (0.0, 6.0)]).astype(np.float64)


def test_knn_dist_default_unchanged():
    assert PHATEModule().model.knn_dist == "euclidean"


def test_knn_dist_reaches_phate_and_drops_pca():
    m = PHATEModule(knn_dist="precomputed_distance", n_landmark=None)
    assert m.model.knn_dist == "precomputed_distance"
    assert m.model.n_pca is None


def test_precomputed_distance_matches_features(blobs):
    kw = dict(random_state=0, neighborhood_size=10, n_landmark=None, n_pca=None, t=10)
    y_feat = np.asarray(PHATEModule(**kw).fit_transform(blobs))
    y_pre = np.asarray(PHATEModule(knn_dist="precomputed_distance", **kw).fit_transform(cdist(blobs, blobs)))
    # same graph, same diffusion and same MDS, so the layouts agree up to solver noise
    d_feat, d_pre = cdist(y_feat, y_feat), cdist(y_pre, y_pre)
    assert np.corrcoef(d_feat.ravel(), d_pre.ravel())[0, 1] > 0.99


def test_non_phate_backend_rejects_knn_dist():
    with pytest.raises(ValueError, match="knn_dist"):
        PHATEModule(knn_dist="precomputed_distance", backend="gpu_phate")
