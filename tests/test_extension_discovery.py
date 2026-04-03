"""Integration tests for the extension discovery mechanism.

Ensures that _discover_extensions() in main.py correctly finds and registers
SearchPathPlugin classes from the `manylatents.extensions` entry-point group.

These tests guard against regressions like moving the discovery call inside
main() (after @hydra.main has already composed configs), which breaks all
extension search paths.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import types
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


# ---------------------------------------------------------------------------
# 1. Entry-point contract: installed extensions are discoverable
# ---------------------------------------------------------------------------


class TestEntryPointDiscovery:
    """Verify the importlib.metadata entry-point group works."""

    def test_entry_point_group_exists(self):
        """The 'manylatents.extensions' group should be queryable (may be empty
        if no extensions are installed, but must not raise)."""
        eps = entry_points(group="manylatents.extensions")
        assert isinstance(eps, (list, types.GeneratorType, type(eps)))

    @pytest.mark.skipif(
        not list(entry_points(group="manylatents.extensions")),
        reason="no extensions installed",
    )
    def test_all_entry_points_load(self):
        """Every declared entry-point must resolve to a loadable class."""
        for ep in entry_points(group="manylatents.extensions"):
            cls = ep.load()
            assert isinstance(cls, type), (
                f"Entry-point {ep.name!r} resolved to {cls!r}, not a class"
            )

    @pytest.mark.skipif(
        not list(entry_points(group="manylatents.extensions")),
        reason="no extensions installed",
    )
    def test_all_entry_points_are_search_path_plugins(self):
        """Every extension entry-point must be a SearchPathPlugin subclass."""
        for ep in entry_points(group="manylatents.extensions"):
            cls = ep.load()
            assert issubclass(cls, SearchPathPlugin), (
                f"Entry-point {ep.name!r} ({cls}) is not a SearchPathPlugin subclass"
            )

    @pytest.mark.skipif(
        not list(entry_points(group="manylatents.extensions")),
        reason="no extensions installed",
    )
    def test_entry_point_plugins_have_manipulate_search_path(self):
        """Each plugin must implement manipulate_search_path (not just inherit stub)."""
        for ep in entry_points(group="manylatents.extensions"):
            cls = ep.load()
            assert "manipulate_search_path" in cls.__dict__, (
                f"{cls.__name__} inherits manipulate_search_path but doesn't override it"
            )


# ---------------------------------------------------------------------------
# 2. _discover_extensions() correctly registers plugins
# ---------------------------------------------------------------------------


class TestDiscoverExtensions:
    """Verify _discover_extensions() registers entry-point plugins with Hydra."""

    def test_discover_extensions_registers_plugins(self):
        """After _discover_extensions(), all entry-point plugins appear in
        Plugins.instance().discover(SearchPathPlugin)."""
        from manylatents.main import _discover_extensions

        _discover_extensions()

        registered = {
            p.__name__
            for p in Plugins.instance().discover(SearchPathPlugin)
        }
        for ep in entry_points(group="manylatents.extensions"):
            cls = ep.load()
            assert cls.__name__ in registered, (
                f"Extension {ep.name!r} ({cls.__name__}) was not registered "
                f"by _discover_extensions(). Registered: {registered}"
            )

    def test_discover_extensions_idempotent(self):
        """Calling _discover_extensions() twice must not double-register."""
        from manylatents.main import _discover_extensions

        _discover_extensions()
        count_before = len(list(Plugins.instance().discover(SearchPathPlugin)))

        _discover_extensions()
        count_after = len(list(Plugins.instance().discover(SearchPathPlugin)))

        assert count_after == count_before

    def test_discover_extensions_tolerates_bad_entry_point(self):
        """If an entry-point fails to load, _discover_extensions() should not
        crash — it silently skips."""
        from manylatents.main import _discover_extensions

        bad_ep = MagicMock()
        bad_ep.load.side_effect = ImportError("broken extension")
        bad_ep.name = "broken"

        # entry_points is imported locally inside _discover_extensions(),
        # so we patch it at the source module
        with patch(
            "importlib.metadata.entry_points",
            return_value=[bad_ep],
        ):
            _discover_extensions()  # must not raise


# ---------------------------------------------------------------------------
# 3. Module-level call: _discover_extensions() runs before @hydra.main
# ---------------------------------------------------------------------------


class TestModuleLevelDiscovery:
    """Guard against the regression where _discover_extensions() was moved
    inside main(), running AFTER @hydra.main had already composed configs."""

    def test_discover_extensions_called_at_module_level(self):
        """main.py must call _discover_extensions() at module level, not
        inside the main() function.

        We verify by AST-parsing main.py: the call should appear as a
        top-level Expr node, not nested inside a FunctionDef.
        """
        main_path = Path(inspect.getfile(importlib.import_module("manylatents.main")))
        source = main_path.read_text()
        tree = ast.parse(source)

        module_level_calls = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == "_discover_extensions":
                    module_level_calls.append(node.lineno)

        assert module_level_calls, (
            "_discover_extensions() is NOT called at module level in main.py. "
            "It MUST run before @hydra.main fires, otherwise extension search "
            "paths are registered too late for config composition."
        )

    def test_discover_extensions_not_called_inside_main(self):
        """_discover_extensions() must NOT be called inside main().

        If it's inside main(), Hydra has already composed configs by the time
        the search paths are registered — extensions can't be found.
        """
        main_path = Path(inspect.getfile(importlib.import_module("manylatents.main")))
        source = main_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        name = None
                        if isinstance(func, ast.Name):
                            name = func.id
                        elif isinstance(func, ast.Attribute):
                            name = func.attr
                        assert name != "_discover_extensions", (
                            f"_discover_extensions() is called INSIDE main() "
                            f"(line {child.lineno}). This is a regression — "
                            f"it must be at module level so it runs before "
                            f"@hydra.main composes configs."
                        )

    def test_discover_before_hydra_main_decorator(self):
        """The _discover_extensions() call must appear BEFORE the @hydra.main
        decorated function in the source file."""
        main_path = Path(inspect.getfile(importlib.import_module("manylatents.main")))
        source = main_path.read_text()
        tree = ast.parse(source)

        discover_line = None
        hydra_main_line = None

        # Find module-level _discover_extensions() call
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "_discover_extensions":
                    discover_line = node.lineno

        # Find the @hydra.main decorated function
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                        if dec.func.attr == "main":
                            hydra_main_line = node.lineno

        assert discover_line is not None, (
            "Could not find module-level _discover_extensions() call"
        )
        assert hydra_main_line is not None, (
            "Could not find @hydra.main decorated function"
        )
        assert discover_line < hydra_main_line, (
            f"_discover_extensions() (line {discover_line}) must appear BEFORE "
            f"@hydra.main (line {hydra_main_line}) in main.py"
        )


# ---------------------------------------------------------------------------
# 4. Omics-specific checks (when installed)
# ---------------------------------------------------------------------------


_has_omics = bool([
    ep for ep in entry_points(group="manylatents.extensions")
    if ep.name == "omics"
])


@pytest.mark.skipif(not _has_omics, reason="manylatents-omics not installed")
class TestOmicsExtension:
    """Verify the omics extension specifically integrates correctly."""

    def test_omics_entry_point_value(self):
        """The omics entry-point must point to OmicsSearchPathPlugin."""
        eps = [
            ep for ep in entry_points(group="manylatents.extensions")
            if ep.name == "omics"
        ]
        assert len(eps) == 1
        assert "OmicsSearchPathPlugin" in eps[0].value

    def test_omics_plugin_registered(self):
        """After discovery, OmicsSearchPathPlugin must be in Hydra's registry."""
        from manylatents.main import _discover_extensions

        _discover_extensions()

        registered_names = {
            p.__name__
            for p in Plugins.instance().discover(SearchPathPlugin)
        }
        assert "OmicsSearchPathPlugin" in registered_names

    def test_omics_plugin_adds_search_paths(self):
        """OmicsSearchPathPlugin.manipulate_search_path must add at least one
        pkg:// path for omics configs."""
        eps = [
            ep for ep in entry_points(group="manylatents.extensions")
            if ep.name == "omics"
        ]
        cls = eps[0].load()
        plugin = cls()

        # Mock a search path and check what gets appended
        mock_search_path = MagicMock()
        plugin.manipulate_search_path(mock_search_path)

        # Should have called append/prepend at least once
        calls = mock_search_path.append.call_args_list + mock_search_path.prepend.call_args_list
        assert calls, (
            "OmicsSearchPathPlugin.manipulate_search_path() did not "
            "append or prepend any search paths"
        )

        # At least one path should reference manylatents configs
        paths = [
            call.kwargs.get("path", call.args[1] if len(call.args) > 1 else "")
            for call in calls
        ]
        assert any("manylatents" in p for p in paths), (
            f"No manylatents config paths registered. Paths: {paths}"
        )

    def test_omics_plugin_module_importable(self):
        """The module containing OmicsSearchPathPlugin must be importable."""
        import manylatents.omics_plugin  # noqa: F401


# ---------------------------------------------------------------------------
# 5. Core plugin still works alongside extensions
# ---------------------------------------------------------------------------


class TestCorePluginCoexistence:
    """Verify the core ManylatentsSearchPathPlugin is not disrupted by
    extension discovery."""

    def test_core_plugin_registered(self):
        """ManylatentsSearchPathPlugin must always be registered."""
        registered_names = {
            p.__name__
            for p in Plugins.instance().discover(SearchPathPlugin)
        }
        assert "ManylatentsSearchPathPlugin" in registered_names

    def test_core_and_extensions_both_registered(self):
        """After discovery, both core and extension plugins must coexist."""
        from manylatents.main import _discover_extensions

        _discover_extensions()

        registered_names = {
            p.__name__
            for p in Plugins.instance().discover(SearchPathPlugin)
        }
        assert "ManylatentsSearchPathPlugin" in registered_names

        # If extensions are installed, they should also be present
        for ep in entry_points(group="manylatents.extensions"):
            cls = ep.load()
            assert cls.__name__ in registered_names
