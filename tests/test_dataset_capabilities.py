"""Tests for dataset capability discovery."""
import numpy as np
import pytest


class FakeDatasetWithAll:
    """Fake dataset with all ground truth methods."""
    def get_gt_dists(self):
        return np.eye(10)

    def get_graph(self):
        return "graph"

    def get_labels(self):
        return np.arange(10)

    def get_centers(self):
        return np.zeros((3, 2))


class FakeDatasetMinimal:
    """Fake dataset with no ground truth."""
    pass


def test_get_capabilities_full_dataset():
    from manylatents.data.capabilities import get_capabilities

    caps = get_capabilities(FakeDatasetWithAll())
    assert caps["gt_dists"] is True
    assert caps["graph"] is True
    assert caps["labels"] is True
    assert caps["centers"] is True


def test_get_capabilities_minimal_dataset():
    from manylatents.data.capabilities import get_capabilities

    caps = get_capabilities(FakeDatasetMinimal())
    assert caps["gt_dists"] is False
    assert caps["graph"] is False
    assert caps["labels"] is False
    assert caps["centers"] is False
    assert caps["gt_type"] == "unknown"


def test_gt_type_manifold():
    """Default gt_type for datasets with gt_dists."""
    from manylatents.data.capabilities import get_capabilities

    class FakeSwissRoll:
        def get_gt_dists(self): return np.eye(5)
        def get_graph(self): return "g"
        def get_labels(self): return np.arange(5)

    caps = get_capabilities(FakeSwissRoll())
    assert caps["gt_type"] == "manifold"


def test_gt_type_graph():
    """gt_type is 'graph' for DLA/Tree datasets."""
    from manylatents.data.capabilities import get_capabilities

    class DLATreeDataset:
        def get_gt_dists(self): return np.eye(5)
        def get_graph(self): return "g"
        def get_labels(self): return np.arange(5)

    caps = get_capabilities(DLATreeDataset())
    assert caps["gt_type"] == "graph"


def test_gt_type_euclidean():
    """gt_type is 'euclidean' for Blob datasets."""
    from manylatents.data.capabilities import get_capabilities

    class BlobsDataset:
        def get_gt_dists(self): return np.eye(5)
        def get_graph(self): return "g"
        def get_labels(self): return np.arange(5)
        def get_centers(self): return np.zeros((3, 2))

    caps = get_capabilities(BlobsDataset())
    assert caps["gt_type"] == "euclidean"


def test_log_capabilities():
    """log_capabilities returns caps dict and doesn't crash."""
    from manylatents.data.capabilities import log_capabilities

    caps = log_capabilities(FakeDatasetWithAll())
    assert isinstance(caps, dict)
    assert "gt_type" in caps
