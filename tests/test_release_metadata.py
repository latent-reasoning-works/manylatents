"""Release metadata: the extras of the one manylatents distribution."""
import datetime as _dt
import re
import subprocess
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
    assert extras["graph"] == ["manykinds[arrays]>=0.1.2"]
    assert extras["all"] == ["manylatents[hf,torchdr,jax,omics,cflows,graph]"]
    assert "manykinds[arrays]>=0.1.2" in extras["cflows"]


def test_manykinds_is_a_core_dependency():
    # manylatents.params re-exports the checks of manykinds.params (standard library only).
    assert "manykinds>=0.1.2" in _pyproject()["project"]["dependencies"]


def test_no_split_distribution():
    data = _pyproject()
    assert "yale" not in data["project"]["optional-dependencies"]
    assert "workspace" not in data.get("tool", {}).get("uv", {})
    assert not (ROOT / "packages" / "manylatents-yale").exists()


# ── version sources and the changelog ───────────────────────────────────────────────────────────


CHANGELOG_NAMES = [
    "max_N", "subsample_indices", "`graph`", "_nodes.csv", "_edges.csv", "_graph.npz",
    "THIRD_PARTY_NOTICES.md", "LicenseRef-Yale-Non-Commercial", "archetypes",
    "manylatents.datamodules", "manylatents[graph]",
]


def version_mismatches(versions: dict) -> list:
    """One problem per source whose version differs from pyproject's."""
    want = versions["pyproject"]
    return [f"{k} says {v}, pyproject says {want}"
            for k, v in sorted(versions.items()) if k != "pyproject" and v != want]


def _headings(text: str) -> list:
    return re.findall(r"^## \[([^\]]+)\](?: - (\S+))?", text, re.MULTILINE)


def changelog_layout_problems(text: str, version: str) -> list:
    """`## [Unreleased]` then `## [<version>] - <ISO date>` as the next heading."""
    heads = _headings(text)
    if len(heads) < 2 or heads[0][0] != "Unreleased":
        return ["the first heading is not ## [Unreleased]"]
    name, date = heads[1]
    if name != version:
        return [f"the heading after Unreleased is [{name}], not [{version}]"]
    try:
        _dt.date.fromisoformat(date)
    except ValueError:
        return [f"[{version}] has no valid ISO date: {date!r}"]
    return []


def _section(text: str, name: str) -> str:
    m = re.search(rf"^## \[{re.escape(name)}\].*?(?=^## \[|\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(0) if m else ""


def unreleased_bullets(text: str) -> list:
    return [ln for ln in _section(text, "Unreleased").splitlines() if ln.startswith("- ")]


def missing_changelog_names(section: str) -> list:
    return [n for n in CHANGELOG_NAMES if n not in section]


def _versions() -> dict:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    entry = [p for p in lock["package"] if p["name"] == "manylatents"]
    cff = re.search(r"^version:\s*(\S+)", (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    return {
        "pyproject": _pyproject()["project"]["version"],
        "lock_manylatents": entry[0]["version"] if entry else None,
        "citation": cff.group(1) if cff else None,
    }


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:3])


@pytest.mark.parametrize("case", ["stale_citation", "missing_name", "plain_graph", "bad_layout"])
def test_helpers_reject_fixture_texts(case):
    if case == "stale_citation":
        problems = version_mismatches(
            {"pyproject": "0.1.9", "citation": "0.1.0", "lock_manylatents": "0.1.9"})
        assert len(problems) == 1 and "citation" in problems[0] and "0.1.0" in problems[0]
    elif case == "missing_name":
        section = " ".join(n for n in CHANGELOG_NAMES if n != "subsample_indices")
        assert missing_changelog_names(section) == ["subsample_indices"]
    elif case == "plain_graph":
        section = " ".join(n for n in CHANGELOG_NAMES if n != "`graph`") + " graph files"
        assert "`graph`" in missing_changelog_names(section.replace("`graph`", ""))
    else:
        text = "## [Unreleased]\n\n## [0.1.9] - 2026-13-40\n"
        assert changelog_layout_problems(text, "0.1.9")
        assert not changelog_layout_problems("## [Unreleased]\n\n## [0.1.9] - 2026-11-03\n", "0.1.9")


def test_version_sources_agree():
    assert version_mismatches(_versions()) == []


def test_changelog_has_section_for_version():
    text = (ROOT / "CHANGELOG.md").read_text()
    assert changelog_layout_problems(text, _versions()["pyproject"]) == []


def test_citation_date_matches_changelog():
    version = _versions()["pyproject"]
    heads = dict(_headings((ROOT / "CHANGELOG.md").read_text()))
    cff = re.search(r"^date-released:\s*['\"]?(\S+?)['\"]?$",
                    (ROOT / "CITATION.cff").read_text(), re.MULTILINE)
    assert cff and heads.get(version) == cff.group(1), (
        f"CHANGELOG [{version}] date {heads.get(version)} vs CITATION.cff "
        f"date-released {cff.group(1) if cff else None}")


def test_changelog_0_1_9_names_both_changes():
    version = _versions()["pyproject"]
    if _version_tuple(version) < (0, 1, 9):
        pytest.skip(f"version {version} is below 0.1.9")
    section = _section((ROOT / "CHANGELOG.md").read_text(), version)
    assert missing_changelog_names(section) == []


def test_unreleased_is_empty_at_release_tag():
    version = _versions()["pyproject"]
    tag = subprocess.run(["git", "describe", "--tags", "--exact-match"], cwd=ROOT, check=False,
                         capture_output=True, text=True).stdout.strip()
    if tag != f"v{version}":
        pytest.skip("not on the release tag")
    assert unreleased_bullets((ROOT / "CHANGELOG.md").read_text()) == []
