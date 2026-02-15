"""Tests for docs/macros.py auto-gen table functions."""
import sys
import os
import pytest

# Add docs/ to path so we can import macros
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs"))
import macros


def test_algorithm_table_latent():
    """algorithm_table('latent') returns a markdown table with known algorithms."""
    table = macros._algorithm_table("latent")
    assert "| algorithm |" in table.lower() or "| Algorithm |" in table
    assert "PCA" in table
    assert "UMAP" in table
    assert "`algorithms/latent=pca`" in table


def test_algorithm_table_lightning():
    """algorithm_table('lightning') returns lightning algorithms."""
    table = macros._algorithm_table("lightning")
    assert "ae_reconstruction" in table or "Reconstruction" in table


def test_metrics_table_embedding():
    """metrics_table('embedding') returns embedding metrics."""
    # Trigger metric registration
    import manylatents.metrics  # noqa: F401

    table = macros._metrics_table("embedding")
    assert "trustworthiness" in table.lower()
    assert "`metrics/embedding=" in table


def test_metrics_table_module():
    """metrics_table('module') returns module metrics."""
    import manylatents.metrics  # noqa: F401

    table = macros._metrics_table("module")
    assert "affinity_spectrum" in table.lower() or "AffinitySpectrum" in table


def test_data_table():
    """data_table() returns data modules."""
    table = macros._data_table()
    assert "swissroll" in table.lower() or "SwissRoll" in table
    assert "`data=" in table


def test_sampling_table():
    """sampling_table() returns sampling strategies."""
    table = macros._sampling_table()
    assert "random" in table.lower() or "Random" in table


def test_tables_are_valid_markdown():
    """All tables should have header separator row."""
    import manylatents.metrics  # noqa: F401

    for fn in [
        lambda: macros._algorithm_table("latent"),
        lambda: macros._metrics_table("embedding"),
        lambda: macros._data_table(),
        lambda: macros._sampling_table(),
    ]:
        table = fn()
        lines = table.strip().split("\n")
        assert len(lines) >= 3, f"Table too short: {table[:100]}"
        assert "---" in lines[1], f"No separator row: {lines[1]}"
