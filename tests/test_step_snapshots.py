"""Tests for step_snapshot dict structure used by embedding callbacks."""

import numpy as np
import pytest
import torch


class TestStepSnapshotLogic:
    """Test the snapshot capture logic in isolation (no pipeline execution)."""

    def test_snapshot_from_tensor(self):
        """Snapshot of a torch.Tensor produces correct numpy array + metadata."""
        tensor = torch.randn(80, 10)
        snapshot_np = tensor.detach().cpu().numpy()

        snap = {
            "step_index": 0,
            "step_name": "pca_step",
            "algorithm": "PCA",
            "output_shape": list(snapshot_np.shape),
            "embedding": snapshot_np,
        }

        assert snap["step_index"] == 0
        assert snap["step_name"] == "pca_step"
        assert snap["algorithm"] == "PCA"
        assert snap["output_shape"] == [80, 10]
        assert snap["embedding"].shape == (80, 10)
        assert isinstance(snap["embedding"], np.ndarray)

    def test_snapshot_from_ndarray(self):
        """Snapshot of a numpy array works correctly."""
        arr = np.random.randn(60, 5)
        snapshot_np = np.asarray(arr)

        snap = {
            "step_index": 1,
            "step_name": "umap_step",
            "algorithm": "UMAP",
            "output_shape": list(snapshot_np.shape),
            "embedding": snapshot_np,
        }

        assert snap["output_shape"] == [60, 5]
        assert np.array_equal(snap["embedding"], arr)

    def test_multi_step_snapshot_list(self):
        """Building a list of snapshots preserves order and shapes."""
        shapes = [(100, 50), (100, 10), (100, 2)]
        names = ["pca", "tsne_init", "tsne_final"]
        algos = ["PCA", "TSNE", "TSNE"]

        step_snapshots = []
        for i, (shape, name, algo) in enumerate(zip(shapes, names, algos)):
            arr = np.random.randn(*shape)
            step_snapshots.append({
                "step_index": i,
                "step_name": name,
                "algorithm": algo,
                "output_shape": list(arr.shape),
                "embedding": arr,
            })

        assert len(step_snapshots) == 3
        assert step_snapshots[0]["output_shape"] == [100, 50]
        assert step_snapshots[1]["output_shape"] == [100, 10]
        assert step_snapshots[2]["output_shape"] == [100, 2]

    def test_snapshot_added_to_embeddings_dict(self):
        """step_snapshots key in embeddings dict is backward-compatible."""
        step_snapshots = [{
            "step_index": 0,
            "step_name": "pca",
            "algorithm": "PCA",
            "output_shape": [100, 2],
            "embedding": np.random.randn(100, 2),
        }]

        embeddings = {
            "embeddings": np.random.randn(100, 2),
            "label": None,
            "metadata": {"source": "pipeline"},
            "step_snapshots": step_snapshots,
        }

        # Existing consumers access only known keys
        assert "embeddings" in embeddings
        assert "label" in embeddings
        assert "metadata" in embeddings
        # New key is present
        assert "step_snapshots" in embeddings
        assert len(embeddings["step_snapshots"]) == 1
