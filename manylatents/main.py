"""
Smart CLI router for both single-algorithm and multi-step pipeline experiments.

This entry point automatically detects the configuration type and routes to
the appropriate execution engine.

Usage (standalone):
    python -m manylatents.main --config-path=manylatents/configs --config-name=config data=swissroll

Usage (via experimentStash):
    # experimentStash provides --config-path automatically
    python scripts/run_experiment manylatents <experiment_name>

Note:
    config_path=None allows CLI override for experimentStash compatibility.
    See experimentStash/CLAUDE.md for the tool contract.
"""

from typing import Dict, Any

import hydra
from omegaconf import DictConfig

# Config registration happens automatically on import
import manylatents.configs
from manylatents.experiment import run_algorithm, run_pipeline

# Import shop utilities (optional - for SLURM job submission and dynamic config discovery)
try:
    import shop  # noqa: F401
    # Register DynamicSearchPathPlugin to discover pkg:// configs via HYDRA_SEARCH_PACKAGES env var
    # This enables experimentStash to specify search packages without hardcoding them here
    from shop.hydra import register_dynamic_search_path
    register_dynamic_search_path()
except ImportError:
    pass  # shop not installed - SLURM launchers and dynamic search path won't be available


@hydra.main(config_path=None, config_name="config", version_base=None)
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
