from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import Config

## Register configs immediately on import
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

## NOTE: SearchPathPlugin registration is now handled centrally by shop's
## DynamicSearchPathPlugin via HYDRA_SEARCH_PACKAGES env var.
## See shop/hydra/search_path.py for details.

__all__ = [
    "Config",
]