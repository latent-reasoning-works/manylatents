from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from .config import Config

## Register configs immediately on import
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


class ManylatentsSearchPathPlugin(SearchPathPlugin):
    """Registers manylatents config paths with Hydra."""

    def manipulate_search_path(self, search_path):
        search_path.append(provider="manylatents", path="pkg://manylatents.configs")


## Entry-points don't work with Hydra 1.3 (it only scans hydra_plugins namespace),
## so we register manually on import.
_plugins = Plugins.instance()
if ManylatentsSearchPathPlugin not in list(_plugins.discover(SearchPathPlugin)):
    _plugins.register(ManylatentsSearchPathPlugin)

__all__ = [
    "Config",
    "ManylatentsSearchPathPlugin",
]