# manylatents/lightning/callbacks/tests/test_wandb_probe.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from manylatents.lightning.callbacks.wandb_probe import WandbProbeLogger


def test_wandb_probe_logger_logs_spread():
    """Should log spread metric to wandb."""
    trajectory = [
        (0, np.eye(10)),
        (100, np.eye(10) * 0.9),
    ]

    with patch("wandb.run", MagicMock()), \
         patch("wandb.log") as mock_log:
        logger = WandbProbeLogger(log_spread=True)
        logger.log_trajectory(trajectory)

        # Should have logged spread
        mock_log.assert_called()
        call_args = mock_log.call_args_list
        logged_keys = set()
        for call in call_args:
            logged_keys.update(call[0][0].keys())

        assert "probe/spread" in logged_keys or any("spread" in k for k in logged_keys)


def test_wandb_probe_logger_logs_operator_stats():
    """Should log operator statistics."""
    trajectory = [
        (0, np.eye(10)),
    ]

    with patch("wandb.run", MagicMock()), \
         patch("wandb.log") as mock_log:
        logger = WandbProbeLogger(log_operator_stats=True)
        logger.log_trajectory(trajectory)

        mock_log.assert_called()


def test_wandb_probe_logger_no_wandb_run():
    """Should silently do nothing if no wandb run."""
    trajectory = [
        (0, np.eye(10)),
    ]

    with patch("wandb.run", None):
        logger = WandbProbeLogger()
        # Should not raise
        logger.log_trajectory(trajectory)


def test_wandb_probe_logger_multi_model():
    """Should log multi-model spread."""
    trajectories = [
        [(0, np.eye(10)), (100, np.eye(10) * 0.9)],
        [(0, np.eye(10) * 1.1), (100, np.eye(10) * 0.95)],
    ]

    with patch("wandb.run", MagicMock()), \
         patch("wandb.log") as mock_log:
        logger = WandbProbeLogger()
        logger.log_multi_model_spread(trajectories)

        mock_log.assert_called()
        call_args = mock_log.call_args_list
        logged_keys = set()
        for call in call_args:
            logged_keys.update(call[0][0].keys())

        assert any("multi_model" in k for k in logged_keys)
