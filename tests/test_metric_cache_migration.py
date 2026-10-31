"""Tests that all kNN-based metrics accept cache=None kwarg and use it."""
import numpy as np
import pytest
from manylatents.utils.metrics import compute_knn


@pytest.fixture
def embedding_data():
    rng = np.random.RandomState(42)
    emb = rng.randn(50, 2).astype(np.float32)
    high_dim = rng.randn(50, 10).astype(np.float32)

    class FakeDataset:
        data = high_dim

    return emb, FakeDataset()


def test_knn_preservation_cache(embedding_data):
    from manylatents.metrics.knn_preservation import KNNPreservation
    emb, ds = embedding_data
    cache = {}
    result = KNNPreservation(emb, ds, n_neighbors=5, cache=cache)
    assert isinstance(result, float)
    assert len(cache) == 2  # both emb and dataset.data cached


def test_lid_cache(embedding_data):
    from manylatents.metrics.lid import LocalIntrinsicDimensionality
    emb, ds = embedding_data
    cache = {}
    result = LocalIntrinsicDimensionality(emb, cache=cache, k=5)
    assert isinstance(result, float)
    assert len(cache) == 1  # only emb cached


def test_trustworthiness_cache(embedding_data):
    from manylatents.metrics.trustworthiness import Trustworthiness
    emb, ds = embedding_data
    cache = {}
    result = Trustworthiness(emb, ds, n_neighbors=5, cache=cache)
    assert isinstance(result, float)


def test_continuity_cache(embedding_data):
    from manylatents.metrics.continuity import Continuity
    emb, ds = embedding_data
    cache = {}
    result = Continuity(emb, ds, n_neighbors=5, cache=cache)
    assert isinstance(result, float)


def test_participation_ratio_cache(embedding_data):
    from manylatents.metrics.local_spectral_analysis import ParticipationRatio
    emb, ds = embedding_data
    cache = {}
    result = ParticipationRatio(emb, cache=cache, n_neighbors=5)
    assert isinstance(result, float)
    assert len(cache) == 1


def test_tangent_space_cache(embedding_data):
    from manylatents.metrics.tangent_space import TangentSpaceApproximation
    emb, ds = embedding_data
    cache = {}
    result = TangentSpaceApproximation(emb, cache=cache, n_neighbors=5)
    assert isinstance(result, float)


def test_cache_sharing_across_metrics(embedding_data):
    """Two metrics sharing same cache don't recompute kNN."""
    from manylatents.metrics.knn_preservation import KNNPreservation
    from manylatents.metrics.lid import LocalIntrinsicDimensionality
    emb, ds = embedding_data

    cache = {}
    # Pre-warm with k=10
    compute_knn(emb, k=10, cache=cache)

    # k<=10, so this reuses the pre-warmed entry
    KNNPreservation(emb, ds, n_neighbors=5, cache=cache)
    assert len(cache) <= 2  # emb + dataset.data from KNNPres

    # LID deliberately does NOT reuse it. Since #304 it RMS-normalises and
    # deduplicates before any distance work, and the cache is keyed by content,
    # so its conditioned view takes its own entry. Reusing the raw entry would
    # reintroduce the bug through the cache: a kNN computed on a badly scaled
    # array is precisely what LID must not trust, and a deduplicated cloud is a
    # different point set. One extra entry is the cost of that.
    LocalIntrinsicDimensionality(emb, k=5, cache=cache)
    assert len(cache) == 3
