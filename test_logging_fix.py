#!/usr/bin/env python3
"""
Test script to verify that manyLatents properly respects logger=None and debug=True.

This tests that when Geomancer calls manyLatents with logger=None, all WandB
operations are disabled and metrics are returned cleanly.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def test_logger_none():
    """Test that logger=None disables all WandB operations."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: logger=None (Geomancer orchestration mode)")
    logger.info("="*70)

    # Import after setting up logging
    from manylatents.api import run

    # Test configuration: logger=None should disable all WandB
    try:
        result = run(
            data='swissroll',
            algorithms={
                'latent': {
                    '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                    'n_components': 2
                }
            },
            metrics={
                'embedding': {
                    'trustworthiness': {},
                    'continuity': {}
                }
            },
            logger=None,  # CRITICAL: Should disable all WandB
            debug=False,
            seed=42
        )

        logger.info("✅ Test passed: logger=None")
        logger.info(f"   Embeddings shape: {result['embeddings'].shape}")
        logger.info(f"   Metrics computed: {list(result.get('scores', {}).keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debug_true():
    """Test that debug=True disables all WandB operations."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: debug=True (fast testing mode)")
    logger.info("="*70)

    from manylatents.api import run

    try:
        result = run(
            data='swissroll',
            algorithms={
                'latent': {
                    '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                    'n_components': 2
                }
            },
            metrics={
                'embedding': {
                    'trustworthiness': {},
                }
            },
            debug=True,  # Should disable all WandB
            seed=42
        )

        logger.info("✅ Test passed: debug=True")
        logger.info(f"   Embeddings shape: {result['embeddings'].shape}")
        logger.info(f"   Metrics computed: {list(result.get('scores', {}).keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wandb_mode_env():
    """Test that WANDB_MODE=disabled environment variable works."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: WANDB_MODE=disabled environment variable")
    logger.info("="*70)

    # Set environment variable
    os.environ['WANDB_MODE'] = 'disabled'

    from manylatents.api import run

    try:
        result = run(
            data='swissroll',
            algorithms={
                'latent': {
                    '_target_': 'manylatents.algorithms.latent.pca.PCAModule',
                    'n_components': 2
                }
            },
            metrics={
                'embedding': {
                    'trustworthiness': {},
                }
            },
            seed=42
        )

        logger.info("✅ Test passed: WANDB_MODE=disabled")
        logger.info(f"   Embeddings shape: {result['embeddings'].shape}")
        logger.info(f"   Metrics computed: {list(result.get('scores', {}).keys())}")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        del os.environ['WANDB_MODE']


def test_geomancer_integration():
    """Test manyLatents called via manyAgents adapter (as Geomancer does)."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Geomancer integration (via manyAgents adapter)")
    logger.info("="*70)

    try:
        # This simulates what Geomancer does
        from manyagents.adapters import ManyLatentsAdapter
        import asyncio

        adapter = ManyLatentsAdapter()

        # Create synthetic data
        data = np.random.randn(100, 50)

        task_config = {
            'algorithm': 'pca',
            'n_components': 2,
            'data': 'swissroll',  # Required for validation, overridden by input_data
            'metrics': {
                'embedding': {
                    'trustworthiness': {},
                    'continuity': {}
                }
            }
        }

        logging_config = {
            'logging_mode': 'collect_only',  # Geomancer collect-only mode
            'save_metrics': True,
            'save_visualizations': False
        }

        # Run via adapter
        result = asyncio.run(adapter.run(
            task_config=task_config,
            input_files={},
            input_data=data,
            logging_config=logging_config
        ))

        if result['success']:
            embeddings = result['output_files']['embeddings']
            scores = result['output_files']['scores']
            logger.info("✅ Test passed: Geomancer integration")
            logger.info(f"   Embeddings shape: {embeddings.shape}")
            logger.info(f"   Metrics computed: {list(scores.keys())}")
            return True
        else:
            logger.error(f"❌ Test failed: Adapter returned success=False")
            logger.error(f"   Error: {result.get('metadata', {}).get('error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    logger.info("Starting manyLatents logging fix tests...")

    results = []

    # Run all tests
    results.append(("logger=None", test_logger_none()))
    results.append(("debug=True", test_debug_true()))
    results.append(("WANDB_MODE=disabled", test_wandb_mode_env()))
    results.append(("Geomancer integration", test_geomancer_integration()))

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed")
        sys.exit(1)
