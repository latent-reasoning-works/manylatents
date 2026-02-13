"""Tests for SilhouetteScore metric."""
import numpy as np
import pytest


def test_silhouette_with_labels():
    """SilhouetteScore returns float when labels available."""
    from manylatents.metrics.silhouette import SilhouetteScore

    rng = np.random.RandomState(42)
    # Two clear clusters
    emb = np.vstack([rng.randn(25, 2) + 5, rng.randn(25, 2) - 5])
    labels = np.array([0] * 25 + [1] * 25)

    class FakeDataset:
        metadata = labels

    result = SilhouetteScore(embeddings=emb, dataset=FakeDataset())
    assert isinstance(result, float)
    assert -1 <= result <= 1
    assert result > 0.5  # Clear clusters -> high score


def test_silhouette_no_labels_returns_nan():
    """SilhouetteScore returns nan when no labels available."""
    from manylatents.metrics.silhouette import SilhouetteScore

    class FakeDataset:
        metadata = None

    result = SilhouetteScore(embeddings=np.zeros((10, 2)), dataset=FakeDataset())
    assert np.isnan(result)


def test_silhouette_no_dataset_returns_nan():
    """SilhouetteScore returns nan when no dataset."""
    from manylatents.metrics.silhouette import SilhouetteScore

    result = SilhouetteScore(embeddings=np.zeros((10, 2)))
    assert np.isnan(result)
