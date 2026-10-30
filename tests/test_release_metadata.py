"""Release metadata: the extras of the one manylatents distribution."""
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    assert ROOT.is_dir()
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_graph_extra_names_only_manykinds():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert extras["graph"] == ["manykinds>=0.1.1"]
    assert extras["all"] == ["manylatents[hf,torchdr,jax,omics,cflows,graph]"]
    assert "manykinds>=0.1.1" in extras["cflows"]


def test_no_split_distribution():
    data = _pyproject()
    assert "yale" not in data["project"]["optional-dependencies"]
    assert "workspace" not in data.get("tool", {}).get("uv", {})
    assert not (ROOT / "packages" / "manylatents-yale").exists()
