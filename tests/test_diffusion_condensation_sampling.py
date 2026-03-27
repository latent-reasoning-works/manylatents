"""Tests for DiffusionCondensationSampling."""
import numpy as np
import pytest

multiscale_phate = pytest.importorskip("multiscale_phate")

from manylatents.utils.sampling import DiffusionCondensationSampling


def _make_blobs(n_per_cluster=100, n_clusters=3, n_features=10, seed=42):
    """Generate well-separated Gaussian blobs with a fake dataset."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, n_features) * 10
    data = np.vstack([
        centers[i] + rng.randn(n_per_cluster, n_features) * 0.3
        for i in range(n_clusters)
    ])

    class _DS:
        pass

    ds = _DS()
    ds.data = data
    embeddings = rng.randn(data.shape[0], 2)
    return embeddings, ds


class TestDiffusionCondensationSampling:
    def test_returns_correct_tuple(self):
        """sample() returns (embeddings, dataset, indices) tuple."""
        embeddings, ds = _make_blobs(n_per_cluster=100, n_clusters=3)
        sampler = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        emb_sub, ds_sub, indices = sampler.sample(embeddings, ds)
        assert isinstance(emb_sub, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert emb_sub.shape[0] == len(indices)
        assert emb_sub.shape[1] == 2
        assert emb_sub.shape[0] <= embeddings.shape[0]

    def test_imbalanced_subsamples(self):
        """Imbalanced clusters get median-truncated, reducing total count."""
        # One big cluster + two small ones -> median truncation should reduce
        rng = np.random.RandomState(42)
        big = rng.randn(200, 10) * 0.3 + 10  # cluster at +10
        small_a = rng.randn(30, 10) * 0.3 - 10  # cluster at -10
        small_b = rng.randn(30, 10) * 0.3  # cluster at 0
        data = np.vstack([big, small_a, small_b])

        class _DS:
            pass

        ds = _DS()
        ds.data = data
        embeddings = rng.randn(data.shape[0], 2)

        sampler = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        emb_sub, _, indices = sampler.sample(embeddings, ds)
        # With imbalanced clusters, median truncation should reduce the total
        assert emb_sub.shape[0] < embeddings.shape[0]

    def test_indices_valid(self):
        """Returned indices are within bounds and sorted."""
        embeddings, ds = _make_blobs(n_per_cluster=100, n_clusters=3)
        sampler = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        _, _, indices = sampler.sample(embeddings, ds)
        assert indices.min() >= 0
        assert indices.max() < embeddings.shape[0]
        np.testing.assert_array_equal(indices, np.sort(indices))

    def test_auto_mode(self):
        """target_clusters=None uses gradient-based salient level."""
        embeddings, ds = _make_blobs(n_per_cluster=100, n_clusters=3)
        sampler = DiffusionCondensationSampling(
            target_clusters=None, landmarks=200, knn=5, seed=42
        )
        emb_sub, _, indices = sampler.sample(embeddings, ds)
        assert emb_sub.shape[0] == len(indices)
        assert emb_sub.shape[0] > 0

    def test_determinism(self):
        """Same seed produces identical indices."""
        embeddings, ds = _make_blobs(n_per_cluster=100, n_clusters=3)
        s1 = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        s2 = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        _, _, idx1 = s1.sample(embeddings, ds)
        _, _, idx2 = s2.sample(embeddings, ds)
        np.testing.assert_array_equal(idx1, idx2)

    def test_dataset_data_subsampled(self):
        """Subsampled dataset .data matches indices count."""
        embeddings, ds = _make_blobs(n_per_cluster=100, n_clusters=3)
        sampler = DiffusionCondensationSampling(
            target_clusters=3, landmarks=200, knn=5, seed=42
        )
        _, ds_sub, indices = sampler.sample(embeddings, ds)
        assert ds_sub.data.shape[0] == len(indices)
