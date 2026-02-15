"""Tests for DatasetTopologyDescriptor metric."""
import numpy as np
import pytest


def test_topology_descriptor_returns_dict():
    """DatasetTopologyDescriptor returns dict with expected keys."""
    from manylatents.metrics.dataset_topology_descriptor import DatasetTopologyDescriptor

    A = np.eye(5) * np.arange(1, 6)
    cache = {"eigenvalues": np.sort(np.diag(A))[::-1]}

    class FakeDataset:
        metadata = np.arange(5)
        def get_gt_dists(self): return np.eye(5)
        def get_labels(self): return np.arange(5)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    result = DatasetTopologyDescriptor(
        embeddings=np.zeros((5, 2)),
        dataset=FakeDataset(),
        module=FakeModule(),
        cache=cache,
    )
    assert isinstance(result, dict)
    assert "spectral_gap" in result
    assert "gt_type" in result
    assert "effective_dim" in result
    assert result["n_samples"] == 5
    assert result["n_features"] == 2


def test_topology_descriptor_no_module():
    """DatasetTopologyDescriptor works without module/cache."""
    from manylatents.metrics.dataset_topology_descriptor import DatasetTopologyDescriptor

    result = DatasetTopologyDescriptor(
        embeddings=np.zeros((10, 3)),
    )
    assert isinstance(result, dict)
    assert result["n_samples"] == 10
    assert result["gt_type"] == "unknown"
    assert np.isnan(result["spectral_gap"])
