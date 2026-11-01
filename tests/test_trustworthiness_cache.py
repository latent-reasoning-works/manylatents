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
    from manylatents.metrics.continuity import Continuity
    from manylatents.metrics.trustworthiness import Trustworthiness
    from manylatents.utils.metrics import compute_knn
    _, low, ds = pair
    cache = {}
    compute_knn(low, k=10, cache=cache)
    compute_knn(ds.data, k=10, cache=cache)
    t = Trustworthiness(low, dataset=ds, n_neighbors=5, cache=cache)
    c = Continuity(low, dataset=ds, n_neighbors=5, cache=cache)
    assert 0 <= t <= 1 and 0 <= c <= 1


def _reference_trustworthiness(high, low, k):
    """The full-matrix form this module used before 0.1.11: cdist, then a double argsort."""
    from scipy.spatial.distance import cdist

    from manylatents.utils.metrics import compute_knn
    n = high.shape[0]
    _, knn_high = compute_knn(high, k=k, include_self=False)
    _, knn_low = compute_knn(low, k=k, include_self=False)
    dist = cdist(high, high)
    np.fill_diagonal(dist, -np.inf)
    rank = np.argsort(np.argsort(dist, axis=1), axis=1)
    penalty = sum(rank[i, j] - k for i in range(n) for j in knn_low[i]
                  if j not in set(knn_high[i]))
    return 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * penalty


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_trustworthiness_equals_the_full_matrix_form(dtype):
    """Chunked distances and direct rank counts give the value of the full rank matrix."""
    from manylatents.metrics.trustworthiness import Trustworthiness
    rng = np.random.RandomState(0)
    high = rng.randn(700, 300).astype(dtype)
    low = (high[:, :2] + 0.5 * rng.randn(700, 2)).astype(dtype)

    class DS:
        data = high
    for k in (5, 25):
        ours = Trustworthiness(low, dataset=DS(), n_neighbors=k)
        ref = _reference_trustworthiness(high.astype(np.float64), low, k)
        assert abs(ours - ref) < 1e-9, (k, ours, ref)

