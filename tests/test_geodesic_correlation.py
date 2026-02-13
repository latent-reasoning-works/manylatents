"""Tests for GeodesicDistanceCorrelation metric."""
import numpy as np
import pytest


def test_geodesic_correlation_spearman():
    """Spearman correlation between ground truth and embedding distances."""
    from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

    rng = np.random.RandomState(42)
    n = 20
    gt_dists = rng.rand(n, n)
    gt_dists = (gt_dists + gt_dists.T) / 2
    np.fill_diagonal(gt_dists, 0)

    class FakeDataset:
        metadata = None
        def get_gt_dists(self):
            return gt_dists

    emb = rng.randn(n, 2)
    result = GeodesicDistanceCorrelation(
        embeddings=emb, dataset=FakeDataset(), correlation_type="spearman"
    )
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_geodesic_correlation_no_gt_returns_nan():
    """Returns nan when dataset has no get_gt_dists."""
    from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

    class FakeDataset:
        metadata = None

    result = GeodesicDistanceCorrelation(embeddings=np.zeros((10, 2)), dataset=FakeDataset())
    assert np.isnan(result)


def test_geodesic_correlation_no_dataset_returns_nan():
    """Returns nan when no dataset provided."""
    from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

    result = GeodesicDistanceCorrelation(embeddings=np.zeros((10, 2)))
    assert np.isnan(result)
