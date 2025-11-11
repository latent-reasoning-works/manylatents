#!/usr/bin/env python
"""
Test PlotEmbeddings callback in offline mode (no WandB upload).

This verifies that:
1. Plots are saved to disk
2. Plot paths are returned in callback_outputs
3. WandB upload is skipped when enable_wandb_upload=False
"""

import logging
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_offline_callback():
    """Test PlotEmbeddings in offline mode."""

    logger.info("=" * 70)
    logger.info("Testing PlotEmbeddings Offline Mode")
    logger.info("=" * 70)

    # Import after path setup
    from manylatents.callbacks.embedding.plot_embeddings import PlotEmbeddings
    from manylatents.data.synthetic_dataset import SwissRoll
    import numpy as np

    # Create test dataset
    dataset = SwissRoll()

    # Create test embeddings (2D)
    n_samples = 1000
    embeddings = {
        'embeddings': np.random.randn(n_samples, 2),
        'label': np.random.randint(0, 3, size=n_samples),
        'metadata': {'algorithm': 'TestAlgo'},
        'scores': {}
    }

    # Test 1: Offline mode (no WandB upload)
    logger.info("\n--- Test 1: Offline Mode ---")
    save_dir = "/tmp/manylatents_offline_test"
    os.makedirs(save_dir, exist_ok=True)

    callback = PlotEmbeddings(
        save_dir=save_dir,
        experiment_name="offline_test",
        enable_wandb_upload=False  # Disable WandB
    )

    result = callback.on_latent_end(dataset=dataset, embeddings=embeddings)

    # Verify results
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'embedding_plot_path' in result, f"Missing plot path in result: {result.keys()}"

    plot_path = result['embedding_plot_path']
    logger.info(f"Plot saved to: {plot_path}")

    # Verify file exists
    assert os.path.exists(plot_path), f"Plot file not found: {plot_path}"
    assert plot_path.endswith('.png'), f"Plot path should be PNG: {plot_path}"

    logger.info("✅ Offline mode test PASSED")

    # Test 2: Online mode (WandB upload enabled but no active run)
    logger.info("\n--- Test 2: Online Mode (no active WandB run) ---")

    callback_online = PlotEmbeddings(
        save_dir=save_dir,
        experiment_name="online_test",
        enable_wandb_upload=True  # Enable WandB (but no run active)
    )

    result_online = callback_online.on_latent_end(dataset=dataset, embeddings=embeddings)

    # Verify results
    assert isinstance(result_online, dict), f"Expected dict, got {type(result_online)}"
    assert 'embedding_plot_path' in result_online, f"Missing plot path in result"

    plot_path_online = result_online['embedding_plot_path']
    logger.info(f"Plot saved to: {plot_path_online}")

    # Verify file exists
    assert os.path.exists(plot_path_online), f"Plot file not found: {plot_path_online}"

    logger.info("✅ Online mode (no WandB run) test PASSED")

    logger.info("\n" + "=" * 70)
    logger.info("All Offline Callback Tests PASSED!")
    logger.info("=" * 70)

    logger.info("\nSummary:")
    logger.info(f"  - Offline mode plot: {plot_path}")
    logger.info(f"  - Online mode plot: {plot_path_online}")
    logger.info(f"  - Both plots exist and paths are returned correctly")


if __name__ == "__main__":
    try:
        test_offline_callback()
        logger.info("\n🎉 Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
