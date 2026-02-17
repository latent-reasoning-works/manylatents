"""
Tests for PyTorch Lightning logger adapters.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

wandb = pytest.importorskip("wandb")
from manylatents.utils.lightning_adapters import WandbRunAdapter


class TestWandbRunAdapter:
    """Test suite for WandbRunAdapter."""
    
    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb.Run object."""
        mock_run = Mock()
        mock_run.name = "test_workflow"
        mock_run.id = "abc123"
        mock_run.config = MagicMock()
        mock_run.log = Mock()
        return mock_run
    
    def test_initialization_without_prefix(self, mock_wandb_run):
        """Test adapter initialization without prefix."""
        adapter = WandbRunAdapter(mock_wandb_run)
        
        assert adapter.name == "test_workflow"
        assert adapter.version == "abc123"
        assert adapter._prefix == ""
        assert adapter.experiment is mock_wandb_run
    
    def test_initialization_with_prefix(self, mock_wandb_run):
        """Test adapter initialization with prefix."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="step_0")
        
        # Should auto-add trailing slash
        assert adapter._prefix == "step_0/"
    
    def test_initialization_with_trailing_slash(self, mock_wandb_run):
        """Test adapter handles prefix with trailing slash."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="step_0/")
        
        assert adapter._prefix == "step_0/"
    
    def test_log_metrics_without_prefix(self, mock_wandb_run):
        """Test logging metrics without prefix."""
        adapter = WandbRunAdapter(mock_wandb_run)
        
        metrics = {"loss": 0.5, "accuracy": 0.95}
        adapter.log_metrics(metrics, step=10)
        
        # Should log metrics as-is
        mock_wandb_run.log.assert_called_once_with(metrics, step=10)
    
    def test_log_metrics_with_prefix(self, mock_wandb_run):
        """Test logging metrics with prefix."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="train/")
        
        metrics = {"loss": 0.5, "accuracy": 0.95}
        adapter.log_metrics(metrics, step=10)
        
        # Should log with prefixed keys
        expected = {"train/loss": 0.5, "train/accuracy": 0.95}
        mock_wandb_run.log.assert_called_once_with(expected, step=10)
    
    def test_log_hyperparams_without_prefix(self, mock_wandb_run):
        """Test logging hyperparameters without prefix."""
        adapter = WandbRunAdapter(mock_wandb_run)
        
        params = {"learning_rate": 0.001, "batch_size": 32}
        adapter.log_hyperparams(params)
        
        # Should update config as-is
        mock_wandb_run.config.update.assert_called_once_with(
            params, 
            allow_val_change=True
        )
    
    def test_log_hyperparams_with_prefix(self, mock_wandb_run):
        """Test logging hyperparameters with prefix."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="model/")
        
        params = {"learning_rate": 0.001, "batch_size": 32}
        adapter.log_hyperparams(params)
        
        # Should update config with prefixed keys
        expected = {"model/learning_rate": 0.001, "model/batch_size": 32}
        mock_wandb_run.config.update.assert_called_once_with(
            expected,
            allow_val_change=True
        )
    
    def test_finalize_does_not_close_run(self, mock_wandb_run):
        """Test that finalize() doesn't close the wrapped run."""
        adapter = WandbRunAdapter(mock_wandb_run)
        
        # Add a finish method to mock
        mock_wandb_run.finish = Mock()
        
        adapter.finalize("success")
        
        # Should NOT call finish() on wrapped run
        mock_wandb_run.finish.assert_not_called()
    
    def test_repr(self, mock_wandb_run):
        """Test string representation."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="step_0/")
        
        repr_str = repr(adapter)
        assert "WandbRunAdapter" in repr_str
        assert "test_workflow" in repr_str
        assert "abc123" in repr_str
        assert "step_0/" in repr_str
    
    def test_integration_with_lightning_trainer(self, mock_wandb_run):
        """Test that adapter works with Lightning Trainer interface."""
        from lightning.pytorch.loggers.logger import Logger
        
        adapter = WandbRunAdapter(mock_wandb_run, prefix="train/")
        
        # Should be a valid Lightning Logger
        assert isinstance(adapter, Logger)
        
        # Should have required properties
        assert hasattr(adapter, "name")
        assert hasattr(adapter, "version")
        assert hasattr(adapter, "experiment")
        
        # Should have required methods
        assert hasattr(adapter, "log_metrics")
        assert hasattr(adapter, "log_hyperparams")
        assert hasattr(adapter, "finalize")


class TestHierarchicalLogging:
    """Test hierarchical logging patterns."""
    
    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb.Run with tracking."""
        mock_run = Mock()
        mock_run.name = "parent_workflow"
        mock_run.id = "parent_123"
        mock_run.config = MagicMock()
        
        # Track all log calls
        mock_run.logged_data = []
        def track_log(data, step=None):
            mock_run.logged_data.append((data, step))
        mock_run.log = track_log
        
        return mock_run
    
    def test_multiple_children_same_run(self, mock_wandb_run):
        """Test multiple child loggers logging to same parent run."""
        # Create two child adapters with different prefixes
        adapter1 = WandbRunAdapter(mock_wandb_run, prefix="step_0/")
        adapter2 = WandbRunAdapter(mock_wandb_run, prefix="step_1/")
        
        # Log from both
        adapter1.log_metrics({"loss": 0.5}, step=0)
        adapter2.log_metrics({"loss": 0.3}, step=0)
        
        # Both should have logged to same run
        assert len(mock_wandb_run.logged_data) == 2
        
        # Check prefixes
        data1, _ = mock_wandb_run.logged_data[0]
        data2, _ = mock_wandb_run.logged_data[1]
        
        assert "step_0/loss" in data1
        assert "step_1/loss" in data2
    
    def test_nested_prefixes(self, mock_wandb_run):
        """Test nested prefix patterns."""
        adapter = WandbRunAdapter(mock_wandb_run, prefix="train/step_0/")
        
        adapter.log_metrics({"loss": 0.5})
        
        # Should support nested prefixes
        data, _ = mock_wandb_run.logged_data[0]
        assert "train/step_0/loss" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
