import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "memory: opt-in subprocess memory measurements")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MANYLATENTS_MEMORY_TESTS") != "1":
        for item in items:
            if "memory" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="set MANYLATENTS_MEMORY_TESTS=1"))
