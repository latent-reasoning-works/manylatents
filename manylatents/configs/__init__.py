from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import Config

## Register configs immediately on import
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

## TODO: Import and register algos, networks, latent modules or datamodules
    
__all__ = [
    "Config",
]