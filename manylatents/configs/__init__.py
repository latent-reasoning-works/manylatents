from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .config import Config

## Register configs immediately on import
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

## Auto-register SearchPathPlugin when configs module is imported
## Entry-points don't work with Hydra 1.3 (it only scans hydra_plugins namespace)
def _register_search_path_plugin():
    """Register ManylatentsSearchPathPlugin with Hydra."""
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    from manylatents.plugins.search_path import ManylatentsSearchPathPlugin

    plugins = Plugins.instance()
    existing = list(plugins.discover(SearchPathPlugin))
    if ManylatentsSearchPathPlugin not in existing:
        plugins.register(ManylatentsSearchPathPlugin)

_register_search_path_plugin()

__all__ = [
    "Config",
]