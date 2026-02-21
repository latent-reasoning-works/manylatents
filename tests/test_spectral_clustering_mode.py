"""Tests for DiffusionMapModule spectral clustering mode."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs

from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule


def _make_blobs_data(n_clusters=3, n_per_cluster=60, random_state=42):
    """Well-separated blobs for spectral clustering tests."""
    X, y = make_blobs(
        n_samples=[n_per_cluster] * n_clusters,
        cluster_std=0.3,
        random_state=random_state,
    )
    return torch.tensor(X, dtype=torch.float32), y


class TestSpectralClusteringMode:

    def test_cluster_mode_returns_labels_shape(self):
        """mode='cluster' transform() returns (N, 1) tensor."""
        X, _ = _make_blobs_data(n_clusters=3)
        m = DiffusionMapModule(mode="cluster", n_clusters=3, knn=10, n_landmark=None)
        m.fit(X)
        result = m.transform(X)
        assert result.shape == (X.shape[0], 1)

    def test_cluster_mode_label_values(self):
        """Labels are integers in [0, n_clusters)."""
        X, _ = _make_blobs_data(n_clusters=3)
        m = DiffusionMapModule(mode="cluster", n_clusters=3, knn=10, n_landmark=None)
        m.fit(X)
        labels = m.predict_labels()
        unique = np.unique(labels)
        assert len(unique) == 3
        assert set(unique) == {0, 1, 2}

    def test_auto_detects_cluster_count(self):
        """n_clusters='auto' detects the correct number from eigenvalue gap."""
        X, _ = _make_blobs_data(n_clusters=4, n_per_cluster=80)
        m = DiffusionMapModule(
            mode="cluster", n_clusters="auto", knn=10, n_landmark=None,
        )
        m.fit(X)
        labels = m.predict_labels()
        detected = len(np.unique(labels))
        # Should detect 4 clusters (allow ±1 for edge cases)
        assert 3 <= detected <= 5, f"Expected ~4 clusters, got {detected}"

    def test_predict_labels_raises_in_embed_mode(self):
        """predict_labels() raises RuntimeError when mode='embed'."""
        X, _ = _make_blobs_data()
        m = DiffusionMapModule(mode="embed", knn=10, n_landmark=None)
        m.fit(X)
        with pytest.raises(RuntimeError, match="mode='cluster'"):
            m.predict_labels()

    def test_embed_mode_unchanged(self):
        """mode='embed' still returns continuous embeddings, not labels."""
        X, _ = _make_blobs_data()
        m = DiffusionMapModule(mode="embed", knn=10, n_landmark=None)
        m.fit(X)
        result = m.transform(X)
        # Embeddings should be n_components wide (default 2), not 1
        assert result.shape == (X.shape[0], 2)
        # Values should be continuous, not integer labels
        assert not torch.all(result == result.round())

    def test_determinism(self):
        """Same seed produces same cluster assignments."""
        X, _ = _make_blobs_data()
        m1 = DiffusionMapModule(
            mode="cluster", n_clusters=3, knn=10, n_landmark=None, random_state=42,
        )
        m1.fit(X)
        labels1 = m1.predict_labels()

        m2 = DiffusionMapModule(
            mode="cluster", n_clusters=3, knn=10, n_landmark=None, random_state=42,
        )
        m2.fit(X)
        labels2 = m2.predict_labels()

        np.testing.assert_array_equal(labels1, labels2)

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            DiffusionMapModule(mode="invalid")
