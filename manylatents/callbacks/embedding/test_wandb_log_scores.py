"""
Co-localized test for WandbLogScores callback.
Tests the specific length mismatch error from the dlatree_phate experiment.
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch

from manylatents.callbacks.embedding.wandb_log_scores import WandbLogScores


@pytest.fixture
def mock_wandb():
    """Fixture to mock wandb.run and wandb.log."""
    with patch('wandb.run') as mock_run, patch('wandb.log') as mock_log:
        mock_run.return_value = Mock()
        mock_run.return_value.log = Mock()
        yield mock_run.return_value, mock_log


def test_length_mismatch_error_fixed(mock_wandb):
    """
    Test that the length mismatch error is fixed.
    This reproduces the exact error from the dlatree_phate experiment.
    """
    mock_run, mock_log = mock_wandb
    
    callback = WandbLogScores(log_summary=True, log_table=True, log_k_curve_table=False)
    
    # Simulate the problematic case from the error
    embeddings = {
        "embeddings": np.random.randn(100, 2),
        "scores": {
            # These are the metrics from the error log
            "dataset.stratification": (0.85, np.random.rand(100)),
            "dataset.gt_preservation": (0.92, np.random.rand(100)),
            "dataset.gt_preservation_far": (0.78, np.random.rand(100)),
            "embedding.knn_preservation": (0.88, np.random.rand(100)),
            
            # Add arrays of different lengths (the actual problem)
            "dataset.stratification__per_sample": np.random.rand(100),  # Length 100
            "dataset.gt_preservation__per_sample": np.random.rand(50),   # Length 50 - DIFFERENT!
            "dataset.gt_preservation_far__per_sample": np.random.rand(75),  # Length 75 - DIFFERENT!
            "embedding.knn_preservation__per_sample": np.random.rand(100),  # Length 100
        },
        "metadata": {"source": "DR"}
    }
    
    dataset = Mock()
    
    # This should NOT raise the "All arrays must be of the same length" error
    result = callback.on_dr_end(dataset, embeddings)
    
    # Verify that wandb.log was called
    assert mock_log.called


def test_component_health_check():
    """
    Smoke test to ensure the component can be instantiated and accepts configs.
    """
    # Test instantiation with different configs
    callback1 = WandbLogScores(log_summary=True, log_table=False, log_k_curve_table=False)
    callback2 = WandbLogScores(log_summary=False, log_table=True, log_k_curve_table=False)
    callback3 = WandbLogScores(log_summary=False, log_table=False, log_k_curve_table=True)
    
    # Verify they were created successfully
    assert isinstance(callback1, WandbLogScores)
    assert isinstance(callback2, WandbLogScores)
    assert isinstance(callback3, WandbLogScores)
    
    # Verify configs were applied
    assert callback1.log_summary is True
    assert callback1.log_table is False
    assert callback2.log_table is True
    assert callback2.log_summary is False