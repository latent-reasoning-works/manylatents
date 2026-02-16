"""CLI entry point for manylatents experiments.

Usage:
    python -m manylatents.main data=swissroll algorithms/latent=pca
    python -m manylatents.main -m experiment=spectral_discrimination cluster=mila resources=cpu_light
"""

from typing import Dict, Any

import hydra
from omegaconf import DictConfig

import manylatents.configs  # noqa: F401 — registers SearchPathPlugin on import
from manylatents.experiment import run_algorithm

# Shop: SLURM launchers and cluster configs (optional)
try:
    import shop  # noqa: F401
    from shop.hydra import register_shop_configs, register_dynamic_search_path
    register_shop_configs()
    register_dynamic_search_path()
except ImportError:
    pass

# Auto-discover extension plugins (e.g. manylatents-omics)
from manylatents.extensions import discover_extensions
discover_extensions()


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    return run_algorithm(cfg)


if __name__ == "__main__":
    main()
