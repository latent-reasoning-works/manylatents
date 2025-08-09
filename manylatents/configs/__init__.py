from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import Config

## doesn't currently import algos, networks, 
## latent modules or datamodules, to be completed.

def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    
__all__ = [
    "Config",
]