import numpy as np
from sklearn.datasets import make_blobs


def test_eigenvalue_effective_rank_returns_expected_keys():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    X, _ = make_blobs(n_samples=100, n_features=5, random_state=42)
    result = ParticipationRatio(X.astype(np.float32), n_neighbors=20, output_mode="full")
    expected_keys = {
        "mean_effective_rank", "std_effective_rank", "mean_top_eigenvalue_ratio",
        "effective_rank", "top_eigenvalue_ratio", "eigenvalues",
    }
    assert set(result.keys()) == expected_keys


def test_eigenvalue_effective_rank_shapes():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    n, d, k = 80, 5, 15
    X, _ = make_blobs(n_samples=n, n_features=d, random_state=42)
    result = ParticipationRatio(X.astype(np.float32), n_neighbors=k, output_mode="full")
    assert result["effective_rank"].shape == (n,)
    assert result["top_eigenvalue_ratio"].shape == (n,)
    assert result["eigenvalues"].shape == (n, min(k, d))


def test_eigenvalue_effective_rank_bounds():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    n, d, k = 100, 5, 20
    X, _ = make_blobs(n_samples=n, n_features=d, centers=1, random_state=42)
    result = ParticipationRatio(X.astype(np.float32), n_neighbors=k, output_mode="full")
    valid = result["effective_rank"] > 0
    assert np.all(result["effective_rank"][valid] >= 1.0)
    assert np.all(result["effective_rank"][valid] <= min(k, d))
    assert np.all(result["top_eigenvalue_ratio"] >= 0.0)
    assert np.all(result["top_eigenvalue_ratio"] <= 1.0)


def test_eigenvalue_effective_rank_2d_embedding():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    X, _ = make_blobs(n_samples=100, n_features=2, centers=1, random_state=42)
    result = ParticipationRatio(X.astype(np.float32), n_neighbors=20, output_mode="full")
    valid = result["effective_rank"] > 0
    assert np.all(result["effective_rank"][valid] <= 2.0)
    assert result["eigenvalues"].shape[1] == 2


def test_eigenvalue_effective_rank_line_has_rank_near_1():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    t = np.linspace(0, 10, 200)
    X = np.column_stack([t, np.zeros_like(t)]).astype(np.float32)
    X += np.random.RandomState(42).randn(*X.shape).astype(np.float32) * 1e-4
    result = ParticipationRatio(X, n_neighbors=20, output_mode="full")
    assert result["mean_effective_rank"] < 1.3


def test_eigenvalue_effective_rank_duplicate_points():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    X = np.zeros((50, 3), dtype=np.float32)
    result = ParticipationRatio(X, n_neighbors=20, output_mode="full")
    assert not np.any(np.isnan(result["effective_rank"]))
    assert not np.any(np.isnan(result["top_eigenvalue_ratio"]))


def test_eigenvalue_effective_rank_uses_cache():
    from manylatents.metrics.participation_ratio import ParticipationRatio
    X, _ = make_blobs(n_samples=50, n_features=3, random_state=42)
    cache = {}
    ParticipationRatio(X.astype(np.float32), n_neighbors=15, output_mode="full", cache=cache)
    assert len(cache) > 0


def test_eigenvalue_effective_rank_registered():
    from manylatents.metrics import list_metrics
    names = list_metrics()
    assert "eigenvalue_effective_rank" in names
    assert "effective_rank" in names


def test_scalar_mode_unchanged():
    """Existing scalar behavior should be preserved."""
    from manylatents.metrics.participation_ratio import ParticipationRatio
    X, _ = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)

    scalar = ParticipationRatio(X.astype(np.float32), n_neighbors=20)
    assert isinstance(scalar, float)

    per_sample = ParticipationRatio(X.astype(np.float32), n_neighbors=20, return_per_sample=True)
    assert isinstance(per_sample, np.ndarray)
    assert per_sample.shape == (100,)
