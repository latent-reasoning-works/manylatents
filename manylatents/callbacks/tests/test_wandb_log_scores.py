"""Tests for WandbLogScores scree plot support."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def mock_wandb():
    """Patch wandb module inside wandb_log_scores."""
    with patch(
        "manylatents.callbacks.embedding.wandb_log_scores.wandb"
    ) as mock:
        mock.run = MagicMock()
        mock.Image = MagicMock(side_effect=lambda fig: f"<Image:{id(fig)}>")
        mock.Table = MagicMock()
        mock.log = MagicMock()
        yield mock


@pytest.fixture()
def callback():
    from manylatents.callbacks.embedding.wandb_log_scores import WandbLogScores

    return WandbLogScores(
        log_summary=True,
        log_table=True,
        log_k_curve_table=False,
        log_scree_plot=True,
    )


def _make_embeddings(scores, tag="embedding"):
    return {"scores": scores, "metadata": {"source": tag}}


class TestScreePlotSplitting:
    """Spectrum keys are routed to scree plots, not the per-sample table."""

    def test_spectrum_excluded_from_table(self, callback, mock_wandb):
        scores = {
            "affinity_spectrum": np.array([1.0, 0.9, 0.5, 0.1]),
            "trustworthiness__per_sample": np.array([0.8, 0.7, 0.6]),
        }
        callback.on_latent_end(None, _make_embeddings(scores))

        # The per-sample table should only contain trustworthiness, not spectrum
        table_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("per_sample_metrics" in str(k) for k in c[0][0])
        ]
        assert len(table_calls) == 1
        logged_key = list(table_calls[0][0][0].keys())[0]
        assert "per_sample_metrics" in logged_key

    def test_scree_plot_logged(self, callback, mock_wandb):
        scores = {
            "affinity_spectrum": np.array([1.0, 0.8, 0.3, 0.05]),
        }
        callback.on_latent_end(None, _make_embeddings(scores))

        scree_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("scree" in str(k) for k in c[0][0])
        ]
        assert len(scree_calls) == 1
        mock_wandb.Image.assert_called_once()

    def test_eigenvalue_key_also_matches(self, callback, mock_wandb):
        scores = {
            "eigenvalue_decay": np.array([1.0, 0.5]),
        }
        callback.on_latent_end(None, _make_embeddings(scores))

        scree_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("scree" in str(k) for k in c[0][0])
        ]
        assert len(scree_calls) == 1

    def test_no_scree_when_disabled(self, mock_wandb):
        from manylatents.callbacks.embedding.wandb_log_scores import WandbLogScores

        cb = WandbLogScores(log_scree_plot=False)
        scores = {"affinity_spectrum": np.array([1.0, 0.5])}
        cb.on_latent_end(None, _make_embeddings(scores))

        scree_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("scree" in str(k) for k in c[0][0])
        ]
        assert len(scree_calls) == 0

    def test_regular_arrays_unaffected(self, callback, mock_wandb):
        """Non-spectrum 1-D arrays still go to the per-sample table."""
        scores = {
            "local_scores": np.array([0.1, 0.2, 0.3]),
        }
        callback.on_latent_end(None, _make_embeddings(scores))

        table_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("per_sample_metrics" in str(k) for k in c[0][0])
        ]
        assert len(table_calls) == 1
        scree_calls = [
            c for c in mock_wandb.log.call_args_list
            if any("scree" in str(k) for k in c[0][0])
        ]
        assert len(scree_calls) == 0
