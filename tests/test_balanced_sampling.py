"""Tests for BalancedLabelSampling (median truncation)."""
import numpy as np
import pytest

from manylatents.utils.sampling import BalancedLabelSampling


class _FakeDataset:
    """Minimal dataset stub with .data and a label attribute."""

    def __init__(self, data, labels, label_attr="population_label"):
        self.data = data
        setattr(self, f"_{label_attr}", labels)

    @property
    def population_label(self):
        return self._population_label


def _make_imbalanced(sizes=(1000, 200, 50), n_features=5, seed=0):
    """Create embeddings, dataset, and labels with given cluster sizes."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n_total = sum(sizes)
    data = rng.standard_normal((n_total, n_features))
    embeddings = rng.standard_normal((n_total, 2))
    ds = _FakeDataset(data, labels)
    return embeddings, ds, labels


class TestBalancedLabelSampling:
    def test_median_truncation_basic(self):
        """A=1000, B=200, C=50 -> median=200, total=450."""
        embeddings, ds, _ = _make_imbalanced(sizes=(1000, 200, 50))
        sampler = BalancedLabelSampling(stratify_by="population_label", seed=42)
        emb_sub, ds_sub, indices = sampler.sample(embeddings, ds)
        assert emb_sub.shape[0] == 450
        assert len(indices) == 450
        assert emb_sub.shape[1] == 2

    def test_equal_clusters_keeps_all(self):
        """Equal-sized clusters: median = cluster size, all kept."""
        embeddings, ds, _ = _make_imbalanced(sizes=(100, 100, 100))
        sampler = BalancedLabelSampling(stratify_by="population_label", seed=42)
        emb_sub, ds_sub, indices = sampler.sample(embeddings, ds)
        assert emb_sub.shape[0] == 300
        assert len(indices) == 300

    def test_determinism(self):
        """Same seed produces identical indices."""
        embeddings, ds, _ = _make_imbalanced(sizes=(1000, 200, 50))
        s1 = BalancedLabelSampling(stratify_by="population_label", seed=7)
        s2 = BalancedLabelSampling(stratify_by="population_label", seed=7)
        _, _, idx1 = s1.sample(embeddings, ds)
        _, _, idx2 = s2.sample(embeddings, ds)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        """Different seeds produce different subsamples for large clusters."""
        embeddings, ds, _ = _make_imbalanced(sizes=(1000, 200, 50))
        s1 = BalancedLabelSampling(stratify_by="population_label", seed=1)
        s2 = BalancedLabelSampling(stratify_by="population_label", seed=2)
        _, _, idx1 = s1.sample(embeddings, ds)
        _, _, idx2 = s2.sample(embeddings, ds)
        assert not np.array_equal(idx1, idx2)

    def test_programmatic_labels(self):
        """Passing labels= directly bypasses dataset attribute lookup."""
        rng = np.random.default_rng(0)
        labels = np.array([0]*500 + [1]*500 + [2]*100)
        embeddings = rng.standard_normal((1100, 2))
        ds = _FakeDataset(rng.standard_normal((1100, 5)), np.zeros(1100))
        sampler = BalancedLabelSampling(seed=42)
        emb_sub, _, indices = sampler.sample(embeddings, ds, labels=labels)
        # median of [100, 500, 500] = 500
        # cluster 2 stays 100, clusters 0 and 1 capped to 500 -> 1100
        assert emb_sub.shape[0] == 1100

    def test_dataset_metadata_subsampled(self):
        """Subsampled dataset has correct .data shape."""
        embeddings, ds, _ = _make_imbalanced(sizes=(1000, 200, 50))
        sampler = BalancedLabelSampling(stratify_by="population_label", seed=42)
        _, ds_sub, indices = sampler.sample(embeddings, ds)
        assert ds_sub.data.shape[0] == 450

    def test_indices_sorted(self):
        """Returned indices are sorted."""
        embeddings, ds, _ = _make_imbalanced(sizes=(1000, 200, 50))
        sampler = BalancedLabelSampling(stratify_by="population_label", seed=42)
        _, _, indices = sampler.sample(embeddings, ds)
        np.testing.assert_array_equal(indices, np.sort(indices))

    def test_two_clusters_even_split(self):
        """Two clusters: big=800, small=200 -> median=500, cap big to 500, keep small 200 -> 700."""
        embeddings, ds, _ = _make_imbalanced(sizes=(800, 200))
        sampler = BalancedLabelSampling(stratify_by="population_label", seed=42)
        emb_sub, _, indices = sampler.sample(embeddings, ds)
        # median of [200, 800] = 500. A capped to 500, B stays 200 -> 700
        assert emb_sub.shape[0] == 700

    def test_fallback_missing_attribute(self):
        """Raises when dataset lacks the attribute and no labels provided."""
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((100, 2))

        class _Bare:
            data = rng.standard_normal((100, 5))

        sampler = BalancedLabelSampling(
            stratify_by="nonexistent_attr", seed=42
        )
        with pytest.raises((AttributeError, ValueError)):
            sampler.sample(embeddings, _Bare())
