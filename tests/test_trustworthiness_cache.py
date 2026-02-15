"""Test Trustworthiness uses shared cache and matches sklearn."""
import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness


@pytest.fixture
def pair():
    rng = np.random.RandomState(42)
    high = rng.randn(50, 10).astype(np.float64)
    low = rng.randn(50, 2).astype(np.float64)
    class DS:
        data = high
    return high, low, DS()


def test_trustworthiness_uses_cache(pair):
    from manylatents.metrics.trustworthiness import Trustworthiness
    _, low, ds = pair
    cache = {}
    result = Trustworthiness(low, dataset=ds, n_neighbors=5, cache=cache)
    assert isinstance(result, float) and 0 <= result <= 1
    assert len(cache) >= 1


def test_trustworthiness_matches_sklearn(pair):
    from manylatents.metrics.trustworthiness import Trustworthiness
    high, low, ds = pair
    k = 10
    ours = Trustworthiness(low, dataset=ds, n_neighbors=k)
    theirs = sk_trustworthiness(high, low, n_neighbors=k)
    np.testing.assert_allclose(ours, theirs, atol=0.05)


def test_trustworthiness_cache_sharing(pair):
    from manylatents.metrics.trustworthiness import Trustworthiness
    from manylatents.metrics.continuity import Continuity
    from manylatents.utils.metrics import compute_knn
    _, low, ds = pair
    cache = {}
    compute_knn(low, k=10, cache=cache)
    compute_knn(ds.data, k=10, cache=cache)
    t = Trustworthiness(low, dataset=ds, n_neighbors=5, cache=cache)
    c = Continuity(low, dataset=ds, n_neighbors=5, cache=cache)
    assert 0 <= t <= 1 and 0 <= c <= 1
