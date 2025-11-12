"""
Smart CLI router for both single-algorithm and multi-step pipeline experiments.

This entry point automatically detects the configuration type and routes to
the appropriate execution engine.

Usage:
    # Single algorithm (automatically detected)
    python -m manylatents.main data=swissroll algorithms/latent=pca

    # Pipeline (automatically detected from pipeline key in config)
    python -m manylatents.main experiment=my_pipeline_config
"""

from typing import Dict, Any

import hydra
from omegaconf import DictConfig

# Config registration happens automatically on import
import manylatents.configs
from manylatents.experiment import run_algorithm, run_pipeline

# Import shop to register custom Hydra launchers
import shop  # noqa: F401


@hydra.main(config_path="../manylatents/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    """
    Smart CLI router that automatically detects and executes the appropriate mode.

    Inspects the configuration for a 'pipeline' key:
    - If present and non-empty: routes to run_pipeline()
    - Otherwise: routes to run_algorithm()

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with keys: embeddings, label, metadata, scores
    """
    # Smart routing: inspect config to determine execution mode
    is_pipeline = hasattr(cfg, 'pipeline') and cfg.pipeline is not None and len(cfg.pipeline) > 0

    if is_pipeline:
        # Route to pipeline engine
        return run_pipeline(cfg)
    else:
        # Route to single algorithm engine
        return run_algorithm(cfg)


if __name__ == "__main__":
    main()
