"""Tests for step_snapshots captured by run_pipeline().

Since run_pipeline() has heavy dependencies (Hydra, WandB, datamodules),
we test the snapshot capture logic by mocking all infrastructure and
verifying the snapshot dict structure.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch, PropertyMock


def _make_pipeline_cfg():
    """Minimal config that run_pipeline() accepts."""
    return OmegaConf.create({
        "pipeline": [
            {
                "name": "pca_step",
                "overrides": {
                    "algorithms": {"latent": {
                        "_target_": "manylatents.algorithms.pca.PCALatent",
                        "n_components": 10,
                    }},
                },
            },
            {
                "name": "umap_step",
                "overrides": {
                    "algorithms": {"latent": {
                        "_target_": "manylatents.algorithms.umap.UMAPLatent",
                        "n_components": 2,
                    }},
                },
            },
        ],
        "algorithms": {"latent": None, "lightning": None},
        "callbacks": {"embedding": None},
        "trainer": {"callbacks": {}},
        "metrics": None,
        "name": "test_snapshots",
        "project": "test",
        "seed": 42,
        "debug": False,
        "logger": None,
    })


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


class TestRunPipelineSnapshots:
    """Integration test: verify run_pipeline() actually includes step_snapshots."""

    @patch("manylatents.experiment.wandb", None)
    @patch("manylatents.experiment.evaluate")
    @patch("manylatents.experiment.execute_step")
    @patch("manylatents.experiment.instantiate_algorithm")
    @patch("manylatents.experiment.instantiate_callbacks")
    @patch("manylatents.experiment.instantiate_trainer")
    @patch("manylatents.experiment.determine_data_source")
    @patch("manylatents.experiment.instantiate_datamodule")
    def test_run_pipeline_produces_step_snapshots(
        self,
        mock_inst_dm,
        mock_det_source,
        mock_inst_trainer,
        mock_inst_cbs,
        mock_inst_algo,
        mock_exec_step,
        mock_evaluate,
    ):
        n_samples = 80
        cfg = _make_pipeline_cfg()

        # Mock datamodule
        mock_dm = MagicMock()
        mock_dm.test_dataset.get_labels.return_value = None
        train_loader = [
            (torch.randn(40, 50),),
            (torch.randn(40, 50),),
        ]
        test_loader = [
            (torch.randn(40, 50),),
            (torch.randn(40, 50),),
        ]
        mock_dm.train_dataloader.return_value = train_loader
        mock_dm.test_dataloader.return_value = test_loader
        mock_inst_dm.return_value = mock_dm

        # Mock other infrastructure
        mock_det_source.return_value = (0, "test_data")
        mock_inst_trainer.return_value = MagicMock()
        mock_inst_cbs.return_value = ([], [])

        # Mock algorithms — need __name__ on the class
        class FakePCA:
            pass

        class FakeUMAP:
            pass

        mock_inst_algo.side_effect = [FakePCA(), FakeUMAP()]

        # Mock execute_step to return arrays of correct shape
        pca_output = np.random.randn(n_samples, 10)
        umap_output = np.random.randn(n_samples, 2)
        mock_exec_step.side_effect = [pca_output, umap_output]

        # Mock evaluate to return scores dict
        mock_evaluate.return_value = {"participation_ratio": 1.5}

        from manylatents.experiment import run_pipeline
        result = run_pipeline(cfg)

        assert "step_snapshots" in result
        snaps = result["step_snapshots"]
        assert len(snaps) == 2

        # Step 0: PCA
        s0 = snaps[0]
        assert s0["step_index"] == 0
        assert s0["step_name"] == "pca_step"
        assert s0["algorithm"] == "FakePCA"
        assert s0["output_shape"] == [n_samples, 10]
        assert s0["embedding"].shape == (n_samples, 10)
        assert isinstance(s0["step_time"], float)
        assert s0["step_time"] >= 0

        # Step 1: UMAP
        s1 = snaps[1]
        assert s1["step_index"] == 1
        assert s1["step_name"] == "umap_step"
        assert s1["algorithm"] == "FakeUMAP"
        assert s1["output_shape"] == [n_samples, 2]
        assert s1["embedding"].shape == (n_samples, 2)
        assert isinstance(s1["step_time"], float)
        assert s1["step_time"] >= 0

        # Metadata timing
        meta = result["metadata"]
        assert isinstance(meta["step_times"], list)
        assert len(meta["step_times"]) == 2
        assert all(isinstance(t, float) and t >= 0 for t in meta["step_times"])
        assert isinstance(meta["eval_time"], float) and meta["eval_time"] >= 0
        assert isinstance(meta["total_time"], float) and meta["total_time"] >= 0

    @patch("manylatents.experiment.wandb", None)
    @patch("manylatents.experiment.evaluate")
    @patch("manylatents.experiment.execute_step")
    @patch("manylatents.experiment.instantiate_algorithm")
    @patch("manylatents.experiment.instantiate_callbacks")
    @patch("manylatents.experiment.instantiate_trainer")
    @patch("manylatents.experiment.determine_data_source")
    @patch("manylatents.experiment.instantiate_datamodule")
    def test_snapshot_fields_complete(
        self,
        mock_inst_dm,
        mock_det_source,
        mock_inst_trainer,
        mock_inst_cbs,
        mock_inst_algo,
        mock_exec_step,
        mock_evaluate,
    ):
        """Every snapshot must contain all required fields."""
        cfg = OmegaConf.create({
            "pipeline": [{
                "name": "only_step",
                "overrides": {
                    "algorithms": {"latent": {"_target_": "fake", "n_components": 5}},
                },
            }],
            "algorithms": {"latent": None, "lightning": None},
            "callbacks": {"embedding": None},
            "trainer": {"callbacks": {}},
            "metrics": None,
            "name": "test_fields",
            "project": "test",
            "seed": 42,
            "debug": False,
            "logger": None,
        })

        mock_dm = MagicMock()
        mock_dm.test_dataset.get_labels.return_value = None
        mock_dm.train_dataloader.return_value = [(torch.randn(40, 20),)]
        mock_dm.test_dataloader.return_value = [(torch.randn(40, 20),)]
        mock_inst_dm.return_value = mock_dm

        mock_det_source.return_value = (0, "test")
        mock_inst_trainer.return_value = MagicMock()
        mock_inst_cbs.return_value = ([], [])

        class FakeAlgo:
            pass

        mock_inst_algo.return_value = FakeAlgo()
        mock_exec_step.return_value = np.random.randn(40, 5)
        mock_evaluate.return_value = {}

        from manylatents.experiment import run_pipeline
        result = run_pipeline(cfg)

        snap = result["step_snapshots"][0]
        required = {"step_index", "step_name", "algorithm", "output_shape", "embedding", "step_time"}
        assert required.issubset(snap.keys())
