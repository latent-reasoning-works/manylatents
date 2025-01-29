from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .configs import Config

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

__all__ = [
    "Config",]