"""
SearchPathPlugin for manylatents - registers config paths with Hydra.

This allows `python -m manylatents.main experiment=...` to work without
explicit --config-path flags.
"""

from hydra.core.global_hydra import GlobalHydra
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins


class ManylatentsSearchPathPlugin(SearchPathPlugin):
    """Registers manylatents config paths with Hydra."""

    def manipulate_search_path(self, search_path):
        # Add manylatents configs to the search path
        # pkg:// prefix tells Hydra to look in the installed package
        search_path.append(
            provider="manylatents",
            path="pkg://manylatents.configs",
        )
