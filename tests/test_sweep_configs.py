"""Tests for sweep config validity."""
import pytest
from pathlib import Path

SWEEP_DIR = Path(__file__).parent.parent / "manylatents" / "configs" / "sweep"


def test_sweep_configs_exist():
    """All sweep config files exist."""
    expected = [
        "dataset_algorithm_grid.yaml",
        "umap_parameter_sensitivity.yaml",
        "phate_parameter_sensitivity.yaml",
        "backend_comparison.yaml",
    ]
    for name in expected:
        assert (SWEEP_DIR / name).exists(), f"Missing sweep config: {name}"


def test_sweep_configs_are_valid_yaml():
    """All sweep configs parse as valid YAML."""
    import yaml

    for path in SWEEP_DIR.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None, f"Empty config: {path.name}"
