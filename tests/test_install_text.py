"""User-facing install text uses uv, not pip.

The uv forms: ``uv add <pkg>`` for a project, ``uv tool install <pkg>`` for a command-line tool,
and ``uv run --with <pkg>`` for a one-off. ``uv pip install`` is a uv command and is allowed.
CHANGELOG.md is skipped: its released entries record what shipped.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

ROOT = Path(__file__).resolve().parents[1]
PIP_INSTALL = re.compile(r"(?<!uv )\bpip install\b")


def _hits(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if PIP_INSTALL.search(line)]


def _python_string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_the_pattern_allows_uv_pip_install_only():
    assert _hits("pip install manylatents")
    assert _hits('raise ImportError("Install with: pip install POT")')
    assert not _hits("uv pip install manylatents")
    assert not _hits("uv add manylatents")


def test_no_pip_install_in_user_text():
    readme = ROOT / "README.md"
    docs = ROOT / "docs"
    package = ROOT / "manylatents"
    assert readme.is_file(), readme
    assert docs.is_dir(), docs
    assert package.is_dir(), package

    found: list[str] = []
    text_files = [readme] + sorted(p for p in docs.rglob("*") if p.suffix in (".md", ".yaml", ".yml", ".py", ".html", ".txt"))
    assert len(text_files) > 1, "docs/ holds no text file"
    for path in text_files:
        if path.name == "CHANGELOG.md":
            continue
        found += [f"{path.relative_to(ROOT)}: {hit}" for hit in _hits(path.read_text())]

    sources = sorted(package.rglob("*.py"))
    assert sources, package
    for path in sources:
        for literal in _python_string_literals(path):
            found += [f"{path.relative_to(ROOT)}: {hit}" for hit in _hits(literal)]

    assert not found, "use uv add, uv tool install or uv run --with:\n" + "\n".join(found)
