"""Tests for SaveTrajectory callback and load_trajectory round-trip."""

import json
import os
import tempfile

import numpy as np
import pytest

from manylatents.callbacks.embedding.save_trajectory import SaveTrajectory, load_trajectory


def _make_embeddings_with_snapshots(n_samples=100):
    """Build an embeddings dict with step_snapshots, mimicking run_pipeline output."""
    return {
        "embeddings": np.random.randn(n_samples, 2),
        "scores": {"participation_ratio": 1.93, "trustworthiness": 0.87},
        "metadata": {"source": "pipeline", "num_steps": 2},
        "step_snapshots": [
            {
                "step_index": 0,
                "step_name": "pca_step",
                "algorithm": "PCA",
                "output_shape": [n_samples, 50],
                "embedding": np.random.randn(n_samples, 50),
                "step_time": 0.123,
            },
            {
                "step_index": 1,
                "step_name": "umap_step",
                "algorithm": "UMAP",
                "output_shape": [n_samples, 2],
                "embedding": np.random.randn(n_samples, 2),
                "step_time": 0.456,
            },
        ],
    }


def _make_single_algorithm_embeddings(n_samples=100):
    """Embeddings dict without step_snapshots (single-algorithm run)."""
    return {
        "embeddings": np.random.randn(n_samples, 2),
        "scores": {"participation_ratio": 1.5},
        "metadata": {
            "source": "pipeline",
            "num_steps": 1,
            "final_algorithm_type": "PCA",
        },
    }


class TestSaveTrajectory:

    def test_writes_manifest_and_npy_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="test_exp")
            embeddings = _make_embeddings_with_snapshots(n_samples=80)

            result = cb.on_latent_end(dataset=None, embeddings=embeddings)

            # Manifest exists
            manifest_path = os.path.join(tmpdir, "trajectory.json")
            assert os.path.exists(manifest_path)

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["version"] == "1.0"
            assert manifest["experiment_name"] == "test_exp"
            assert len(manifest["steps"]) == 2
            assert manifest["final_shape"] == [80, 2]
            assert "timestamp" in manifest

            # Per-step NPY files exist with correct shapes
            for step in manifest["steps"]:
                npy_path = os.path.join(tmpdir, step["embedding_file"])
                assert os.path.exists(npy_path)
                arr = np.load(npy_path)
                assert list(arr.shape) == step["output_shape"]

    def test_manifest_step_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="meta_test")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            s0 = manifest["steps"][0]
            assert s0["step_index"] == 0
            assert s0["step_name"] == "pca_step"
            assert s0["algorithm"] == "PCA"
            assert s0["embedding_file"] == "step_0_embedding.npy"

            s1 = manifest["steps"][1]
            assert s1["step_index"] == 1
            assert s1["step_name"] == "umap_step"
            assert s1["algorithm"] == "UMAP"

    def test_scores_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="scores")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert manifest["scores"]["participation_ratio"] == pytest.approx(1.93)
            assert manifest["scores"]["trustworthiness"] == pytest.approx(0.87)

    def test_tuple_scores_flattened(self):
        """Scores stored as (scalar, array) tuples should use the scalar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="tuples")
            embeddings = _make_embeddings_with_snapshots()
            embeddings["scores"]["knn_recall"] = (0.95, np.ones(100))

            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert manifest["scores"]["knn_recall"] == pytest.approx(0.95)

    def test_single_algorithm_fallback(self):
        """Without step_snapshots, should save final embedding as single-step trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="single")
            embeddings = _make_single_algorithm_embeddings(n_samples=60)

            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert len(manifest["steps"]) == 1
            step = manifest["steps"][0]
            assert step["step_name"] == "single_step"
            assert step["algorithm"] == "PCA"
            assert step["embedding_file"] == "step_0_embedding.npy"

            arr = np.load(os.path.join(tmpdir, "step_0_embedding.npy"))
            assert arr.shape == (60, 2)

    def test_registers_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="reg")
            embeddings = _make_embeddings_with_snapshots()

            result = cb.on_latent_end(dataset=None, embeddings=embeddings)

            assert "trajectory_manifest" in result
            assert result["trajectory_manifest"].endswith("trajectory.json")

    def test_empty_scores(self):
        """Should handle embeddings with no scores gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="no_scores")
            embeddings = _make_embeddings_with_snapshots()
            embeddings["scores"] = {}

            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert manifest["scores"] == {}

    def test_saves_label_when_present(self):
        """Label array should be written as label.npy and referenced in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="label_test")
            embeddings = _make_embeddings_with_snapshots(n_samples=50)
            embeddings["label"] = np.arange(50)

            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert manifest["label_file"] == "label.npy"
            label = np.load(os.path.join(tmpdir, "label.npy"))
            np.testing.assert_array_equal(label, np.arange(50))

    def test_label_file_null_when_absent(self):
        """Manifest should have label_file: null when no label is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="no_label")
            embeddings = _make_embeddings_with_snapshots()

            cb.on_latent_end(dataset=None, embeddings=embeddings)

            with open(os.path.join(tmpdir, "trajectory.json")) as f:
                manifest = json.load(f)

            assert manifest["label_file"] is None


class TestLoadTrajectory:

    def test_round_trip_with_snapshots(self):
        """Write via SaveTrajectory, read via load_trajectory, verify contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="round_trip")
            embeddings = _make_embeddings_with_snapshots(n_samples=80)
            embeddings["label"] = np.arange(80)

            cb.on_latent_end(dataset=None, embeddings=embeddings)
            result = load_trajectory(tmpdir)

            assert len(result) == 2

    def test_each_element_has_embedding_outputs_keys(self):
        """Every element should have embeddings, label, metadata, scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="keys_test")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            for step in result:
                assert "embeddings" in step
                assert "label" in step
                assert "metadata" in step
                assert "scores" in step
                assert isinstance(step["embeddings"], np.ndarray)

    def test_embedding_shapes_match(self):
        """Loaded embedding shapes should match what was written."""
        n = 60
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="shapes")
            embeddings = _make_embeddings_with_snapshots(n_samples=n)
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            assert result[0]["embeddings"].shape == (n, 50)  # PCA step
            assert result[1]["embeddings"].shape == (n, 2)   # UMAP step

    def test_label_round_trip_present(self):
        """Label should round-trip correctly when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="label_rt")
            embeddings = _make_embeddings_with_snapshots(n_samples=40)
            expected_label = np.array([0, 1, 2, 3] * 10)
            embeddings["label"] = expected_label

            cb.on_latent_end(dataset=None, embeddings=embeddings)
            result = load_trajectory(tmpdir)

            for step in result:
                np.testing.assert_array_equal(step["label"], expected_label)

    def test_label_none_when_absent(self):
        """Label should be None when not saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="no_label")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            for step in result:
                assert step["label"] is None

    def test_scores_only_on_last_step(self):
        """Scores should only appear on the last step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="scores_last")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            # First step: no scores
            assert result[0]["scores"] == {}
            # Last step: has scores
            assert result[-1]["scores"]["participation_ratio"] == pytest.approx(1.93)
            assert result[-1]["scores"]["trustworthiness"] == pytest.approx(0.87)

    def test_metadata_has_required_fields(self):
        """Each step's metadata should have step_index, step_name, algorithm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="meta")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)

            assert result[0]["metadata"]["step_index"] == 0
            assert result[0]["metadata"]["step_name"] == "pca_step"
            assert result[0]["metadata"]["algorithm"] == "PCA"

            assert result[1]["metadata"]["step_index"] == 1
            assert result[1]["metadata"]["step_name"] == "umap_step"
            assert result[1]["metadata"]["algorithm"] == "UMAP"

    def test_step_time_round_trip(self):
        """step_time should round-trip through SaveTrajectory → manifest → load_trajectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="timing_rt")
            embeddings = _make_embeddings_with_snapshots()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)

            assert result[0]["metadata"]["step_time"] == pytest.approx(0.123)
            assert result[1]["metadata"]["step_time"] == pytest.approx(0.456)

    def test_step_time_none_for_single_algorithm(self):
        """Single-algorithm fallback should have step_time=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="timing_none")
            embeddings = _make_single_algorithm_embeddings()
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            assert result[0]["metadata"]["step_time"] is None

    def test_single_algorithm_round_trip(self):
        """Single-algorithm run should round-trip as a one-element list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = SaveTrajectory(save_dir=tmpdir, experiment_name="single_rt")
            embeddings = _make_single_algorithm_embeddings(n_samples=50)
            cb.on_latent_end(dataset=None, embeddings=embeddings)

            result = load_trajectory(tmpdir)
            assert len(result) == 1
            assert result[0]["embeddings"].shape == (50, 2)
            assert result[0]["metadata"]["algorithm"] == "PCA"
            assert result[0]["scores"]["participation_ratio"] == pytest.approx(1.5)
