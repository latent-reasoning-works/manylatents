# tests/callbacks/test_loadings_analysis.py
"""Tests for LoadingsAnalysisCallback."""
from dataclasses import dataclass
from typing import Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback


@dataclass
class MockChannelLoadings:
    """Mock loadings object returned by get_loadings()."""
    channel_ranges: Dict[str, tuple]
    components: Optional[np.ndarray]
    explained_variance_ratio: Optional[np.ndarray] = None


class MockModule:
    """Mock module with get_loadings() method."""

    def __init__(self, loadings: MockChannelLoadings):
        self._loadings = loadings

    def get_loadings(self) -> MockChannelLoadings:
        return self._loadings


class TestLoadingsAnalysisCallback:
    """Tests for LoadingsAnalysisCallback."""

    def test_init_default_params(self):
        """Test default initialization."""
        callback = LoadingsAnalysisCallback()
        assert callback.modality_dims is None
        assert callback.modality_names is None
        assert callback.threshold == 0.1
        assert callback.log_to_wandb is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        callback = LoadingsAnalysisCallback(
            modality_dims=[1920, 256, 1536],
            modality_names=["dna", "rna", "protein"],
            threshold=0.2,
            log_to_wandb=False,
        )
        assert callback.modality_dims == [1920, 256, 1536]
        assert callback.modality_names == ["dna", "rna", "protein"]
        assert callback.threshold == 0.2
        assert callback.log_to_wandb is False

    def test_compute_modality_contributions(self):
        """Test L2 norm contribution computation."""
        callback = LoadingsAnalysisCallback()

        # Create loadings: 10 features (5+5), 3 components
        loadings = np.array([
            [1.0, 0.0, 0.5],  # modality_a features
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],  # modality_b features
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
        ])
        modality_ranges = {"a": (0, 5), "b": (5, 10)}

        contributions = callback._compute_modality_contributions(loadings, modality_ranges)

        assert "a" in contributions
        assert "b" in contributions
        assert len(contributions["a"]) == 3
        assert len(contributions["b"]) == 3

        # Component 0: modality_a has high contribution, modality_b has none
        assert contributions["a"][0] > 0
        assert contributions["b"][0] == 0

        # Component 1: modality_b has high contribution, modality_a has none
        assert contributions["a"][1] == 0
        assert contributions["b"][1] > 0

        # Component 2: both modalities contribute equally
        assert np.isclose(contributions["a"][2], contributions["b"][2])

    def test_classify_components_all_shared(self):
        """Test classification when all components are shared."""
        callback = LoadingsAnalysisCallback(threshold=0.1)

        # Both modalities contribute equally to all components
        contributions = {
            "a": np.array([1.0, 1.0, 1.0]),
            "b": np.array([1.0, 1.0, 1.0]),
        }

        result = callback._classify_components(contributions)

        assert result["n_shared"] == 3
        assert result["n_specific"] == 0
        assert result["shared_fraction"] == 1.0

    def test_classify_components_all_specific(self):
        """Test classification when all components are modality-specific."""
        callback = LoadingsAnalysisCallback(threshold=0.2)

        # Components alternate between modalities (one dominates completely)
        contributions = {
            "a": np.array([1.0, 0.0, 1.0]),
            "b": np.array([0.0, 1.0, 0.0]),
        }

        result = callback._classify_components(contributions)

        assert result["n_shared"] == 0
        assert result["n_specific"] == 3
        assert result["shared_fraction"] == 0.0

    def test_classify_components_mixed(self):
        """Test classification with mixed shared/specific components."""
        callback = LoadingsAnalysisCallback(threshold=0.1)

        # Component 0: shared (both contribute)
        # Component 1: specific to a (b is 0)
        # Component 2: shared (both contribute)
        contributions = {
            "a": np.array([1.0, 1.0, 0.5]),
            "b": np.array([1.0, 0.0, 0.5]),
        }

        result = callback._classify_components(contributions)

        assert result["n_shared"] == 2
        assert result["n_specific"] == 1
        assert result["shared_fraction"] == pytest.approx(2 / 3)

    def test_on_latent_end_no_module(self):
        """Test callback returns empty dict when no module provided."""
        callback = LoadingsAnalysisCallback()

        result = callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=None,
        )

        assert result == {}

    def test_on_latent_end_module_without_loadings(self):
        """Test callback returns empty dict when module has no get_loadings."""
        callback = LoadingsAnalysisCallback()

        class ModuleWithoutLoadings:
            pass

        result = callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=ModuleWithoutLoadings(),
        )

        assert result == {}

    def test_on_latent_end_with_module(self):
        """Test full callback with mock module."""
        callback = LoadingsAnalysisCallback(
            modality_dims=[5, 5],
            modality_names=["a", "b"],
            threshold=0.1,
            log_to_wandb=False,
        )

        # Create loadings with shared component
        loadings = np.array([
            [1.0, 0.0],  # a features
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],  # b features
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])

        mock_loadings = MockChannelLoadings(
            channel_ranges={"a": (0, 5), "b": (5, 10)},
            components=loadings,
        )
        module = MockModule(mock_loadings)

        result = callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=module,
        )

        assert "n_shared" in result
        assert "n_specific" in result
        assert "shared_fraction" in result
        assert "a_mean_contribution" in result
        assert "b_mean_contribution" in result

    def test_on_latent_end_with_explained_variance(self):
        """Test callback includes shared_variance_ratio when EVR available."""
        callback = LoadingsAnalysisCallback(
            modality_dims=[5, 5],
            modality_names=["a", "b"],
            threshold=0.1,
            log_to_wandb=False,
        )

        # All shared components
        loadings = np.ones((10, 3))
        evr = np.array([0.5, 0.3, 0.2])

        mock_loadings = MockChannelLoadings(
            channel_ranges={"a": (0, 5), "b": (5, 10)},
            components=loadings,
            explained_variance_ratio=evr,
        )
        module = MockModule(mock_loadings)

        result = callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=module,
        )

        assert "shared_variance_ratio" in result
        assert result["shared_variance_ratio"] == pytest.approx(1.0)

    def test_on_latent_end_registers_outputs(self):
        """Test callback registers outputs."""
        callback = LoadingsAnalysisCallback(
            modality_dims=[5, 5],
            modality_names=["a", "b"],
            log_to_wandb=False,
        )

        loadings = np.ones((10, 2))
        mock_loadings = MockChannelLoadings(
            channel_ranges={"a": (0, 5), "b": (5, 10)},
            components=loadings,
        )
        module = MockModule(mock_loadings)

        callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=module,
        )

        # Outputs are stored in callback_outputs attribute
        outputs = callback.callback_outputs
        assert "n_shared" in outputs
        assert "n_specific" in outputs

    def test_three_modalities(self):
        """Test with three modalities (DNA, RNA, Protein)."""
        callback = LoadingsAnalysisCallback(
            modality_dims=[10, 5, 8],  # DNA, RNA, Protein
            modality_names=["dna", "rna", "protein"],
            threshold=0.1,
            log_to_wandb=False,
        )

        # Create loadings: 23 features, 4 components
        # Component 0: all modalities (shared)
        # Component 1: DNA only
        # Component 2: RNA + Protein
        # Component 3: all modalities (shared)
        n_features = 10 + 5 + 8
        loadings = np.zeros((n_features, 4))

        # Shared component 0
        loadings[:, 0] = 1.0

        # DNA-specific component 1
        loadings[:10, 1] = 1.0

        # RNA + Protein component 2
        loadings[10:, 2] = 1.0

        # Shared component 3
        loadings[:, 3] = 0.5

        mock_loadings = MockChannelLoadings(
            channel_ranges={"dna": (0, 10), "rna": (10, 15), "protein": (15, 23)},
            components=loadings,
        )
        module = MockModule(mock_loadings)

        result = callback.on_latent_end(
            dataset=None,
            embeddings={},
            module=module,
        )

        # Components 0 and 3 are shared (all modalities >= threshold)
        # Components 1 and 2 are specific (not all modalities contribute)
        assert result["n_shared"] == 2
        assert result["n_specific"] == 2
        assert result["shared_fraction"] == 0.5

        # Check per-modality stats
        assert "dna_mean_contribution" in result
        assert "rna_mean_contribution" in result
        assert "protein_mean_contribution" in result
