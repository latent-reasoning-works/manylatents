"""Tests for PlotEmbeddings callback."""

import os
import tempfile
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

try:
    import wandb as _wandb
    _wandb.init  # verify real package
    _has_wandb = True
except (ImportError, AttributeError):
    _has_wandb = False

# No-op decorator when wandb is not installed (patch target doesn't exist)
_disable_wandb = patch("wandb.run", None) if _has_wandb else lambda f: f

from manylatents.callbacks.embedding.base import ColormapInfo, ColormapProvider
from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings
from matplotlib.colors import ListedColormap


class MockDataset:
    """Mock dataset without colormap support."""

    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def get_labels(self, col: str = None) -> np.ndarray:
        return self._labels


class MockDatasetWithColormap:
    """Mock dataset implementing ColormapProvider protocol."""

    def __init__(self, labels: np.ndarray, cmap_info: ColormapInfo):
        self._labels = labels
        self._cmap_info = cmap_info

    def get_labels(self, col: str = None) -> np.ndarray:
        return self._labels

    def get_colormap_info(self) -> ColormapInfo:
        return self._cmap_info


class TestPlotEmbeddingsInit:
    """Tests for PlotEmbeddings initialization."""

    def test_init_default_params(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)
            assert callback.legend is False
            assert callback.color_by_score is None
            assert callback.alpha == 0.8
            assert callback.enable_wandb_upload is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(
                save_dir=tmpdir,
                legend=True,
                alpha=0.5,
                color_by_score="test_score",
                enable_wandb_upload=False,
            )
            assert callback.legend is True
            assert callback.alpha == 0.5
            assert callback.color_by_score == "test_score"
            assert callback.enable_wandb_upload is False

    def test_init_creates_save_dir(self):
        """Test that initialization creates the save directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "new_subdir")
            callback = PlotEmbeddings(save_dir=subdir)
            assert os.path.exists(subdir)


class TestGetColormap:
    """Tests for _get_colormap method."""

    def test_metric_provided_viz_metadata(self):
        """Test that metric-provided __viz metadata is used for coloring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, color_by_score="tangent_space")

            # Simulate metric providing viz metadata via __viz key
            embeddings = {
                "scores": {
                    "tangent_space": np.array([1, 2, 1, 2]),
                    "tangent_space__viz": ColormapInfo(
                        cmap="categorical",
                        label_format="Dim = {}",
                        is_categorical=True,
                    ),
                }
            }

            cmap_info = callback._get_colormap(
                MockDataset(np.array([0, 1])), embeddings=embeddings
            )

            assert cmap_info.is_categorical is True
            assert cmap_info.cmap == "categorical"
            assert cmap_info.label_format == "Dim = {}"

    def test_continuous_score_colormap(self):
        """Test continuous score coloring returns viridis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, color_by_score="some_metric")
            cmap_info = callback._get_colormap(MockDataset(np.array([0, 1])))

            assert cmap_info.cmap == "viridis"
            assert cmap_info.is_categorical is False

    def test_dataset_provided_colormap(self):
        """Test dataset with ColormapProvider is used."""
        custom_cmap = ColormapInfo(
            cmap={"A": "red", "B": "blue"},
            label_names={0: "Label A", 1: "Label B"},
            is_categorical=True,
        )
        dataset = MockDatasetWithColormap(np.array(["A", "B"]), custom_cmap)

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)
            cmap_info = callback._get_colormap(dataset)

            assert cmap_info.cmap == {"A": "red", "B": "blue"}
            assert cmap_info.label_names == {0: "Label A", 1: "Label B"}

    def test_fallback_colormap(self):
        """Test fallback to viridis for basic dataset."""
        dataset = MockDataset(np.array([0, 1, 2]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)
            cmap_info = callback._get_colormap(dataset)

            assert cmap_info.cmap == "viridis"
            assert cmap_info.is_categorical is True

    def test_user_override_takes_precedence(self):
        """Test that user overrides take precedence over metric-declared info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(
                save_dir=tmpdir,
                color_by_score="tangent_space",
                cmap_override="plasma",
                is_categorical_override=False,
            )

            # Even with metric providing categorical info, user override wins
            embeddings = {
                "scores": {
                    "tangent_space": np.array([1, 2, 1, 2]),
                    "tangent_space__viz": ColormapInfo(
                        cmap="categorical",
                        is_categorical=True,
                    ),
                }
            }

            cmap_info = callback._get_colormap(
                MockDataset(np.array([0, 1])), embeddings=embeddings
            )

            assert cmap_info.cmap == "plasma"
            assert cmap_info.is_categorical is False


class TestGetEmbeddings:
    """Tests for _get_embeddings method."""

    def test_no_skip_first_element(self):
        """Verify [1:] skip is removed - all elements returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {"embeddings": np.array([[1, 2], [3, 4], [5, 6]])}
            result = callback._get_embeddings(embeddings)

            # Should return all 3 rows, not 2
            assert result.shape == (3, 2)
            np.testing.assert_array_equal(result[0], [1, 2])

    def test_truncates_to_2d(self):
        """Test high-dimensional embeddings are truncated to 2D."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {"embeddings": np.random.randn(10, 50)}
            result = callback._get_embeddings(embeddings)

            assert result.shape == (10, 2)

    def test_handles_torch_tensor(self):
        """Test conversion from torch tensor."""
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {"embeddings": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
            result = callback._get_embeddings(embeddings)

            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 2)


class TestGetColorArray:
    """Tests for _get_color_array method."""

    def test_no_skip_first_element(self):
        """Verify [1:] skip is removed for color array."""
        dataset = MockDataset(np.array([0, 1, 2, 3]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {"label": np.array([0, 1, 2, 3])}
            result = callback._get_color_array(dataset, embeddings)

            # Should return all 4 elements
            assert len(result) == 4

    def test_uses_embeddings_label(self):
        """Test embeddings['label'] takes precedence over dataset."""
        dataset = MockDataset(np.array([10, 20, 30]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {"label": np.array([0, 1, 2])}
            result = callback._get_color_array(dataset, embeddings)

            np.testing.assert_array_equal(result, [0, 1, 2])

    def test_fallback_to_dataset_labels(self):
        """Test fallback to dataset.get_labels() when no embeddings label."""
        dataset = MockDataset(np.array([10, 20, 30]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            embeddings = {}
            result = callback._get_color_array(dataset, embeddings)

            np.testing.assert_array_equal(result, [10, 20, 30])

    def test_color_by_score_from_scores_dict(self):
        """Test color_by_score retrieves from scores dict."""
        dataset = MockDataset(np.array([0, 1]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, color_by_score="my_metric")

            embeddings = {"scores": {"my_metric": np.array([0.5, 0.8])}}
            result = callback._get_color_array(dataset, embeddings)

            np.testing.assert_array_equal(result, [0.5, 0.8])

    def test_color_by_score_fallback_to_label(self):
        """Test color_by_score falls back to label if metric not found."""
        dataset = MockDataset(np.array([0, 1]))

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, color_by_score="missing_metric")

            embeddings = {"label": np.array([5, 6])}
            result = callback._get_color_array(dataset, embeddings)

            np.testing.assert_array_equal(result, [5, 6])


class TestApplyDictColormap:
    """Tests for _apply_dict_colormap method."""

    def test_apply_dict_colormap(self):
        """Test dict colormap maps labels to colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            labels = np.array([1, 2, 1, 3])
            cmap_dict = {1: "red", 2: "blue", 3: "green"}

            colors = callback._apply_dict_colormap(labels, cmap_dict)

            assert colors == ["red", "blue", "red", "green"]

    def test_apply_dict_colormap_missing_label(self):
        """Test dict colormap handles missing labels with gray."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir)

            labels = np.array([1, 99])  # 99 not in dict
            cmap_dict = {1: "red"}

            colors = callback._apply_dict_colormap(labels, cmap_dict)

            assert colors[0] == "red"
            assert colors[1] == "#808080"  # Default gray


class TestOnLatentEnd:
    """Integration tests for full callback execution."""

    @_disable_wandb
    def test_full_pipeline(self):
        """Test complete callback execution."""
        dataset = MockDataset(np.array([0, 1, 0, 1]))
        embeddings = {
            "embeddings": np.array([[0, 0], [1, 1], [0.5, 0.5], [1.5, 1.5]]),
            "label": np.array([0, 1, 0, 1]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, enable_wandb_upload=False)

            result = callback.on_latent_end(dataset, embeddings)

            # Verify file was created
            assert "embedding_plot_path" in result
            assert os.path.exists(result["embedding_plot_path"])
            assert result["embedding_plot_path"].endswith(".png")

    @_disable_wandb
    def test_full_pipeline_with_legend(self):
        """Test callback with legend enabled."""
        dataset = MockDataset(np.array([0, 1, 2]))
        embeddings = {
            "embeddings": np.array([[0, 0], [1, 1], [2, 2]]),
            "label": np.array([0, 1, 2]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, legend=True, enable_wandb_upload=False)

            result = callback.on_latent_end(dataset, embeddings)

            assert "embedding_plot_path" in result
            assert os.path.exists(result["embedding_plot_path"])

    @_disable_wandb
    def test_full_pipeline_with_colormap_provider(self):
        """Test callback with dataset implementing ColormapProvider."""
        custom_cmap = ColormapInfo(
            cmap={0: "#ff0000", 1: "#00ff00"},
            label_names={0: "Red Class", 1: "Green Class"},
            is_categorical=True,
        )
        dataset = MockDatasetWithColormap(np.array([0, 1, 0, 1]), custom_cmap)
        embeddings = {
            "embeddings": np.array([[0, 0], [1, 1], [0.5, 0.5], [1.5, 1.5]]),
            "label": np.array([0, 1, 0, 1]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, legend=True, enable_wandb_upload=False)

            result = callback.on_latent_end(dataset, embeddings)

            assert "embedding_plot_path" in result
            assert os.path.exists(result["embedding_plot_path"])


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_callback_outputs_dict_returned(self):
        """Test that on_latent_end returns callback_outputs dict."""
        dataset = MockDataset(np.array([0, 1]))
        embeddings = {
            "embeddings": np.array([[0, 0], [1, 1]]),
            "label": np.array([0, 1]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = PlotEmbeddings(save_dir=tmpdir, enable_wandb_upload=False)

            if _has_wandb:
                with patch("wandb.run", None):
                    result = callback.on_latent_end(dataset, embeddings)
            else:
                result = callback.on_latent_end(dataset, embeddings)

            assert isinstance(result, dict)
            assert "embedding_plot_path" in result

    def test_overridable_get_colormap(self):
        """Test that _get_colormap can be overridden in subclasses."""

        class CustomPlotEmbeddings(PlotEmbeddings):
            def _get_colormap(self, dataset):
                return ColormapInfo(
                    cmap={"custom": "purple"},
                    label_names={0: "Custom Label"},
                    is_categorical=True,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = CustomPlotEmbeddings(save_dir=tmpdir)
            cmap_info = callback._get_colormap(MockDataset(np.array([0])))

            assert cmap_info.cmap == {"custom": "purple"}
            assert cmap_info.label_names == {0: "Custom Label"}
