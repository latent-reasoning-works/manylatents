"""Auto-discover installed manylatents extension plugins via entry-points.

Extensions declare a SearchPathPlugin class in pyproject.toml:

    [project.entry-points."manylatents.extensions"]
    omics = "manylatents.omics_plugin:OmicsSearchPathPlugin"

Core discovers and registers these at startup before @hydra.main() fires.
"""

import logging
from importlib.metadata import entry_points

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

log = logging.getLogger(__name__)


def discover_extensions():
    """Scan 'manylatents.extensions' entry-point group, register plugins."""
    plugins = Plugins.instance()
    existing = {type(p) for p in plugins.discover(SearchPathPlugin)}

    for ep in entry_points(group="manylatents.extensions"):
        try:
            plugin_cls = ep.load()
            if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, SearchPathPlugin)):
                log.warning(
                    "Extension %r: %s is not a SearchPathPlugin subclass, skipping",
                    ep.name, plugin_cls,
                )
                continue
            if plugin_cls not in existing:
                plugins.register(plugin_cls)
                log.debug("Registered extension: %s (%s)", ep.name, plugin_cls.__name__)
        except Exception as e:
            log.warning("Failed to load extension %r: %s", ep.name, e)
