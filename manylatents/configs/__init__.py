from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import Config

## Register configs immediately on import
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

## SearchPathPlugin is registered via entry_points in pyproject.toml
## See manylatents/plugins/search_path.py

__all__ = [
    "Config",
]