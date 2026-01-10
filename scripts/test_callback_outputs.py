#!/usr/bin/env python
"""
Test that callback outputs are properly returned from run_algorithm.

This verifies that:
1. Callbacks are executed
2. Callback outputs are captured
3. Callback outputs are included in the returned embeddings dict
"""

import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_callback_outputs():
    """Test that callback outputs are properly returned."""

    logger.info("=" * 70)
    logger.info("Testing Callback Outputs Return")
    logger.info("=" * 70)

    from hydra import initialize, compose
    from manylatents.experiment import run_algorithm

    # Initialize Hydra
    with initialize(version_base=None, config_path="../manylatents/configs"):
        # Create config with plot_embeddings callback in offline mode
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=PCA",
                "dataset=swissroll",
                "callbacks/embedding=plot_embeddings",  # Use plot_embeddings callback
                "logger=none",  # Disable WandB (use 'none', not 'null')
                "debug=true",
                "callbacks.embedding.plot_embeddings.enable_wandb_upload=false",  # Enable offline mode
                "callbacks.embedding.plot_embeddings.save_dir=/tmp/callback_test",
            ]
        )

        logger.info("Running algorithm with offline PlotEmbeddings callback...")
        result = run_algorithm(cfg)

        # Verify result structure
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        logger.info(f"Result keys: {list(result.keys())}")

        # Check for callback_outputs
        assert 'callback_outputs' in result, f"Missing callback_outputs in result: {result.keys()}"

        callback_outputs = result['callback_outputs']
        logger.info(f"Callback outputs: {list(callback_outputs.keys())}")

        # Verify plot path is in callback outputs
        assert 'embedding_plot_path' in callback_outputs, f"Missing plot path in callback_outputs: {callback_outputs.keys()}"

        plot_path = callback_outputs['embedding_plot_path']
        logger.info(f"Plot saved to: {plot_path}")

        # Verify file exists
        import os
        assert os.path.exists(plot_path), f"Plot file not found: {plot_path}"

        logger.info("\n" + "=" * 70)
        logger.info("✅ Callback Outputs Test PASSED!")
        logger.info("=" * 70)

        logger.info("\nSummary:")
        logger.info(f"  - Callback executed successfully")
        logger.info(f"  - Plot path returned in callback_outputs: {plot_path}")
        logger.info(f"  - Plot file exists on disk")


if __name__ == "__main__":
    try:
        test_callback_outputs()
        logger.info("\n🎉 Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
