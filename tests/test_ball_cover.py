"""Tests for manylatents.utils.charts.ball_cover."""
import numpy as np
import pytest

from manylatents.utils.charts import BallCover, ball_cover


def naive_ball_cover(X, m):
    """Reference: brute-force neighbours and per-point distance recomputation."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    m = min(m, n)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    nn = np.argsort(D, axis=1, kind="stable")[:, :m]
    covered = np.zeros(n, dtype=bool)
    landmarks = [0]
    covered[nn[0]] = True
    while not covered.all():
        best, best_score = -1, -1.0
        for i in range(n):
            if covered[i]:
                continue
            score = min(((X[i] - X[lm]) ** 2).sum() for lm in landmarks)
            if score > best_score:
                best, best_score = i, score
        landmarks.append(best)
        covered[nn[best]] = True
    return np.array(landmarks), nn[landmarks]


@pytest.mark.parametrize(
    "n,d,m,seed",
    [(50, 2, 5, 0), (120, 3, 8, 1), (200, 10, 15, 2), (80, 5, 2, 3), (300, 4, 30, 4)],
)
def test_matches_naive_reference(n, d, m, seed):
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    got = ball_cover(X, m)
    ref_lms, ref_members = naive_ball_cover(X, m)
    assert isinstance(got, BallCover)
    np.testing.assert_array_equal(got.landmarks, ref_lms)
    np.testing.assert_array_equal(got.members, ref_members)


def test_cover_properties():
    X = np.random.default_rng(5).normal(size=(150, 3))
    bc = ball_cover(X, 7)
    assert bc.members.shape == (len(bc.landmarks), 7)
    np.testing.assert_array_equal(bc.members[:, 0], bc.landmarks)
    assert bc.landmarks[0] == 0
    assert len(np.unique(bc.landmarks)) == len(bc.landmarks)
    assert np.array_equal(np.unique(bc.members), np.arange(150))


def test_deterministic():
    X = np.random.default_rng(6).normal(size=(100, 4))
    a, b = ball_cover(X, 6), ball_cover(X, 6)
    np.testing.assert_array_equal(a.landmarks, b.landmarks)
    np.testing.assert_array_equal(a.members, b.members)


def test_shared_cache_matches_no_cache_and_is_reused(monkeypatch):
    X = np.random.default_rng(7).normal(size=(200, 5)).astype(np.float32)
    ms = [20, 10, 5, 2]
    cache = {}
    ball_cover(X, max(ms), cache=cache)
    assert len(cache) == 1
    (stored_k, _, _), = cache.values()
    assert stored_k == max(ms) - 1
    fresh = {m: ball_cover(X, m) for m in ms}

    # Any further neighbour search would now fail: the cache must serve every m.
    import sklearn.neighbors

    def _fail(*args, **kwargs):
        raise AssertionError("neighbour search ran despite a primed cache")

    monkeypatch.setattr(sklearn.neighbors, "NearestNeighbors", _fail)
    for m in ms:
        got = ball_cover(X, m, cache=cache)
        np.testing.assert_array_equal(got.landmarks, fresh[m].landmarks)
        np.testing.assert_array_equal(got.members, fresh[m].members)
    assert len(cache) == 1


def test_m_equals_one_every_point_is_a_landmark():
    X = np.random.default_rng(8).normal(size=(30, 2))
    bc = ball_cover(X, 1)
    assert bc.landmarks[0] == 0
    assert sorted(bc.landmarks.tolist()) == list(range(30))
    np.testing.assert_array_equal(bc.members[:, 0], bc.landmarks)
    assert bc.members.shape == (30, 1)
    ref_lms, _ = naive_ball_cover(X, 1)
    np.testing.assert_array_equal(bc.landmarks, ref_lms)


@pytest.mark.parametrize("m", [40, 41, 100])
def test_m_at_least_n_gives_one_ball(m):
    X = np.random.default_rng(9).normal(size=(40, 3))
    bc = ball_cover(X, m)
    np.testing.assert_array_equal(bc.landmarks, [0])
    assert bc.members.shape == (1, 40)
    assert bc.members[0, 0] == 0
    assert sorted(bc.members[0].tolist()) == list(range(40))


@pytest.mark.parametrize("m", [1, 3])
def test_single_point(m):
    bc = ball_cover(np.array([[1.0, 2.0]]), m)
    np.testing.assert_array_equal(bc.landmarks, [0])
    np.testing.assert_array_equal(bc.members, [[0]])


def test_float32_and_float64_agree():
    X = np.random.default_rng(10).normal(size=(120, 4)).astype(np.float32)
    a, b = ball_cover(X, 6), ball_cover(X.astype(np.float64), 6)
    np.testing.assert_array_equal(a.landmarks, b.landmarks)
    np.testing.assert_array_equal(a.members, b.members)
    assert a.landmarks.dtype == np.intp


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_m_raises(bad):
    with pytest.raises(ValueError):
        ball_cover(np.zeros((5, 2)), bad)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        ball_cover(np.zeros(5), 2)
    with pytest.raises(ValueError):
        ball_cover(np.zeros((0, 2)), 2)
