"""The Yale non-commercial boundary: per-file headers, NOTICE and THIRD_PARTY_NOTICES.md.

Nine files in this repository are ported from KrishnaswamyLab code that Yale University
publishes under a non-commercial license. These tests keep the file headers, the NOTICE
list, the THIRD_PARTY_NOTICES.md lists and the docs in agreement.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "manylatents"
NOTICES = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
NOTICE = REPO_ROOT / "NOTICE"

YALE_DERIVED_FILES = frozenset(
    {
        "manylatents/algorithms/cflows_granger.py",
        "manylatents/algorithms/lightning/losses/cflows.py",
        "manylatents/algorithms/lightning/cflows.py",
        "manylatents/algorithms/lightning/mioflow.py",
        "manylatents/algorithms/lightning/networks/mioflow_net.py",
        "manylatents/algorithms/lightning/networks/gaga_net.py",
        "manylatents/algorithms/lightning/networks/aanet.py",
        "manylatents/configs/algorithms/lightning/mioflow.yaml",
        "manylatents/configs/algorithms/lightning/mioflow_gaga.yaml",
    }
)

SECTION_OF = {
    "manylatents/algorithms/cflows_granger.py": "KrishnaswamyLab/cflows",
    "manylatents/algorithms/lightning/losses/cflows.py": "KrishnaswamyLab/cflows",
    "manylatents/algorithms/lightning/cflows.py": "KrishnaswamyLab/cflows",
    "manylatents/algorithms/lightning/mioflow.py": "KrishnaswamyLab/MIOFlow",
    "manylatents/algorithms/lightning/networks/mioflow_net.py": "KrishnaswamyLab/MIOFlow",
    "manylatents/algorithms/lightning/networks/gaga_net.py": "KrishnaswamyLab/MIOFlow",
    "manylatents/algorithms/lightning/networks/aanet.py": "KrishnaswamyLab/AAnet",
    "manylatents/configs/algorithms/lightning/mioflow.yaml": "KrishnaswamyLab/MIOFlow",
    "manylatents/configs/algorithms/lightning/mioflow_gaga.yaml": "KrishnaswamyLab/MIOFlow",
}

YALE_SECTIONS = (
    "KrishnaswamyLab/MIOFlow",
    "KrishnaswamyLab/cflows",
    "KrishnaswamyLab/AAnet",
)

HEADER = (
    "# SPDX-License-Identifier: LicenseRef-Yale-Non-Commercial",
    "# See THIRD_PARTY_NOTICES.md at the repository root.",
)
YALE_ID = "LicenseRef-Yale-Non-Commercial"

EXCLUSION_SENTENCE = "The Apache-2.0 license in LICENSE does not apply to the files listed in this document."
ARCHETYPES_SENTENCE = (
    "`aa.py` is licensed under Apache-2.0; the BSD-3-Clause terms below cover only the two "
    "functions adapted from archetypes."
)

# Each upstream licence text is copied verbatim into a fenced block. The digest is the
# SHA-256 of the upstream file with one trailing newline removed, so any edit to the copy
# (reflow, quote or symbol change, whitespace) fails the test.
UPSTREAM = {
    "KrishnaswamyLab/MIOFlow": (
        "https://github.com/KrishnaswamyLab/MIOFlow",
        "4e79ada3ce96feeb3f51ba02c7d999e095eaf7e8",
        "8f7d9760a4bc19a3ec18d0560778c3c3096c50fe24dcd1aa02a28323c25815d8",
    ),
    "KrishnaswamyLab/cflows": (
        "https://github.com/KrishnaswamyLab/cflows",
        "3878169cfa15582d0ebb54abacf1ea1e293e3912",
        "8f7d9760a4bc19a3ec18d0560778c3c3096c50fe24dcd1aa02a28323c25815d8",
    ),
    "KrishnaswamyLab/AAnet": (
        "https://github.com/KrishnaswamyLab/AAnet",
        "6dc3814419572d3295bdd32c8c62b8e6a81e2bbb",
        "4203fefdcef852cbc8cdb0519a5038e7e511ac5690a433009f5fe6f43853af93",
    ),
    "archetypes": (
        "https://github.com/aleixalcacer/archetypes",
        "4518807340135213dbcb94b3512621b28c6eca41",
        "176266725af338c678d5cae5a3984e2d869d01b57aabdb85dd1f077a4a6f118f",
    ),
}

DOCS_URL = "https://github.com/latent-reasoning-works/manylatents/blob/main/THIRD_PARTY_NOTICES.md"


# --- helpers ---------------------------------------------------------------------------


def path_difference(
    found: set[str] | frozenset[str], expected: set[str] | frozenset[str]
) -> str:
    """Describe how ``found`` differs from ``expected``; empty when the sets are equal."""
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if extra:
        parts.append(f"not in YALE_DERIVED_FILES: {extra}")
    return "; ".join(parts)


def notices_sections() -> dict[str, str]:
    """Split THIRD_PARTY_NOTICES.md into its ``## `` sections, keyed by heading text."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in NOTICES.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def derived_files(section: str) -> list[str]:
    """Backticked paths in the ``### Derived files`` list of one section."""
    paths: list[str] = []
    inside = False
    for line in section.splitlines():
        if line.startswith("#"):
            inside = line.strip() == "### Derived files"
            continue
        if inside and line.lstrip().startswith(("- ", "* ")):
            paths.extend(re.findall(r"`([^`]+)`", line))
    return paths


def fenced_block(section: str) -> str:
    """The text inside the first fenced code block of a section."""
    match = re.search(
        r"^```[^\n]*\n(.*?)\n```$", section, flags=re.DOTALL | re.MULTILINE
    )
    assert match, "no fenced block in section"
    return match.group(1)


def notice_excluded_paths() -> set[str]:
    """Paths listed under "Not covered by the Apache-2.0 relicense" in NOTICE."""
    lines = NOTICE.read_text(encoding="utf-8").splitlines()
    start = lines.index("Not covered by the Apache-2.0 relicense:") + 1
    paths: set[str] = set()
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        if line.strip():
            paths.add(line.strip())
    return paths


def files_with_yale_header() -> set[str]:
    assert PACKAGE_ROOT.is_dir()
    found = set()
    for path in [*PACKAGE_ROOT.rglob("*.py"), *PACKAGE_ROOT.rglob("*.yaml")]:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline().rstrip("\n")
        if first == HEADER[0]:
            found.add(path.relative_to(REPO_ROOT).as_posix())
    return found


# --- headers ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", sorted(YALE_DERIVED_FILES))
def test_header_on_each_yale_derived_file(rel_path):
    lines = (REPO_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    assert tuple(lines[:2]) == HEADER


def test_no_other_file_carries_the_yale_identifier():
    assert PACKAGE_ROOT.is_dir()
    offenders = []
    for path in [*PACKAGE_ROOT.rglob("*.py"), *PACKAGE_ROOT.rglob("*.yaml")]:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in YALE_DERIVED_FILES:
            continue
        if YALE_ID in path.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert not offenders, offenders


def test_headers_and_notices_list_the_same_files():
    found = files_with_yale_header()
    listed = {
        p for name in YALE_SECTIONS for p in derived_files(notices_sections()[name])
    }
    message = path_difference(found, listed)
    assert not message, message


def test_mioflow_docstring_survives_the_header():
    from manylatents.algorithms.lightning import mioflow

    assert mioflow.__doc__.startswith("MIOFlow LightningModule training wrapper.")


@pytest.mark.parametrize("name", ["mioflow.yaml", "mioflow_gaga.yaml"])
def test_config_still_loads_after_the_header(name):
    from omegaconf import OmegaConf

    config_dir = PACKAGE_ROOT / "configs" / "algorithms" / "lightning"
    assert config_dir.is_dir()
    cfg = OmegaConf.load(config_dir / name)
    assert cfg._target_ == "manylatents.algorithms.lightning.mioflow.MIOFlow"
    assert (
        cfg.network._target_
        == "manylatents.algorithms.lightning.networks.mioflow_net.MIOFlowODEFunc"
    )


# --- THIRD_PARTY_NOTICES.md ------------------------------------------------------------


def test_notices_has_exactly_the_expected_sections():
    assert set(notices_sections()) == {*YALE_SECTIONS, "archetypes"}


def test_every_derived_path_is_listed_and_exists():
    sections = notices_sections()
    listed = [p for name in YALE_SECTIONS for p in derived_files(sections[name])]
    assert len(listed) == len(set(listed)), "a path is listed twice"
    message = path_difference(set(listed), YALE_DERIVED_FILES)
    assert not message, message
    for rel_path in listed:
        assert (REPO_ROOT / rel_path).is_file(), rel_path
    for rel_path, section in SECTION_OF.items():
        assert rel_path in derived_files(sections[section]), (rel_path, section)
    assert derived_files(sections["KrishnaswamyLab/AAnet"]) == [
        "manylatents/algorithms/lightning/networks/aanet.py"
    ]


def test_apache_exclusion_sentence_is_present():
    assert EXCLUSION_SENTENCE in NOTICES.read_text(encoding="utf-8")


@pytest.mark.parametrize("section_name", sorted(UPSTREAM))
def test_licence_text_is_the_verbatim_upstream_text(section_name):
    url, commit, digest = UPSTREAM[section_name]
    section = notices_sections()[section_name]
    assert url in section
    assert commit in section
    block = fenced_block(section)
    assert hashlib.sha256(block.encode("utf-8")).hexdigest() == digest


@pytest.mark.parametrize("section_name", YALE_SECTIONS)
def test_yale_sections_name_the_licence(section_name):
    section = notices_sections()[section_name]
    assert "Yale" in section and "Non-Commercial" in section
    assert "### Derived files" in section


def test_third_party_notices_has_archetypes_entry():
    section = notices_sections()["archetypes"]
    assert "BSD-3-Clause" in section
    # The verbatim upstream LICENSE spells the line with a comma after the year.
    assert "Copyright (c) 2021, Aleix Alcacer Sales" in section
    assert "manylatents/algorithms/latent/aa.py" in section
    assert "Redistribution and use in source and binary forms" in section
    assert "### Derived files" not in section
    assert ARCHETYPES_SENTENCE in section


def test_comparison_reports_differences():
    new_port = "manylatents/algorithms/lightning/networks/new_port.py"
    message = path_difference(set(YALE_DERIVED_FILES) | {new_port}, YALE_DERIVED_FILES)
    assert new_port in message

    aanet = "manylatents/algorithms/lightning/networks/aanet.py"
    message = path_difference(set(YALE_DERIVED_FILES) - {aanet}, YALE_DERIVED_FILES)
    assert "aanet.py" in message

    assert path_difference(YALE_DERIVED_FILES, YALE_DERIVED_FILES) == ""


# --- NOTICE ----------------------------------------------------------------------------


def test_notice_covers_the_nine_files():
    excluded = notice_excluded_paths()
    message = path_difference(excluded & YALE_DERIVED_FILES, YALE_DERIVED_FILES)
    assert YALE_DERIVED_FILES <= excluded, message


def test_notice_names_each_upstream():
    text = NOTICE.read_text(encoding="utf-8")
    for upstream in YALE_SECTIONS:
        assert upstream in text
    assert "These files implement MIOFlow" not in text


# --- packaging and imports -------------------------------------------------------------


def test_no_split():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "workspace" not in pyproject.get("tool", {}).get("uv", {})
    assert "yale" not in pyproject["project"].get("optional-dependencies", {})
    assert pyproject["project"]["license"] == "Apache-2.0"
    assert not (REPO_ROOT / "packages" / "manylatents-yale").exists()


def test_core_imports_do_not_load_yale_modules():
    modules = sorted(
        p.removesuffix(".py").replace("/", ".")
        for p in YALE_DERIVED_FILES
        if p.endswith(".py")
    )
    code = (
        "import sys\n"
        "import manylatents, manylatents.algorithms.latent, manylatents.api\n"
        f"loaded = [m for m in {modules!r} if m in sys.modules]\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_names_import_from_current_paths():
    code = (
        "from manylatents.algorithms.lightning.mioflow import MIOFlow\n"
        "from manylatents.algorithms.lightning.cflows import Cflows\n"
        "from manylatents.algorithms.lightning.networks.aanet import Vanilla\n"
        "from manylatents.algorithms.lightning.networks.gaga_net import GAGANetwork\n"
        "from manylatents.algorithms.lightning.losses.cflows import OTLoss\n"
        "from manylatents.algorithms.cflows_granger import granger_grn\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


# --- docs ------------------------------------------------------------------------------


def test_docs_section_names_the_boundary():
    text = (REPO_ROOT / "docs" / "algorithms.md").read_text(encoding="utf-8")
    assert "## Licensing of Yale-derived modules" in text
    assert "Yale non-commercial" in text
    assert DOCS_URL in text
    for rel_path in YALE_DERIVED_FILES:
        assert rel_path in text, rel_path


def test_readme_names_the_licence():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Yale non-commercial" in text
    assert "THIRD_PARTY_NOTICES.md" in text
