"""Unit tests for runners.sequential_study."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from manylatents.runners.sequential_study import (
    CellState,
    StudyState,
    run_study,
)


def _make_specs(tmp_path: Path, n: int = 3) -> List[Path]:
    paths = []
    for i in range(n):
        p = tmp_path / f"cell_{i}.json"
        p.write_text(json.dumps({"i": i}))
        paths.append(p)
    return paths


def test_all_cells_succeed(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 3)
    seen: List[Path] = []

    def run_one(p: Path) -> None:
        seen.append(p)

    state_path = tmp_path / "state.json"
    res = run_study(specs, run_one=run_one, state_path=state_path)

    assert seen == specs
    assert [p.name for p in res.completed] == ["cell_0.json", "cell_1.json", "cell_2.json"]
    assert res.failed == []
    assert res.skipped == []
    assert res.pending == []


def test_run_one_receives_path(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 1)
    received: List[Path] = []

    def run_one(p: Path) -> None:
        received.append(p)

    run_study(specs, run_one=run_one, state_path=tmp_path / "s.json")
    assert len(received) == 1
    assert isinstance(received[0], Path)
    assert received[0] == specs[0]


def test_skip_on_failure_continues(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 3)

    def run_one(p: Path) -> None:
        if p.name == "cell_1.json":
            raise RuntimeError("boom")

    res = run_study(specs, run_one=run_one, state_path=tmp_path / "s.json", on_failure="skip")
    assert [p.name for p in res.completed] == ["cell_0.json", "cell_2.json"]
    assert [p.name for p in res.failed] == ["cell_1.json"]


def test_halt_on_failure_stops(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 3)
    seen: List[Path] = []

    def run_one(p: Path) -> None:
        seen.append(p)
        if p.name == "cell_1.json":
            raise RuntimeError("boom")

    res = run_study(specs, run_one=run_one, state_path=tmp_path / "s.json", on_failure="halt")
    assert [p.name for p in seen] == ["cell_0.json", "cell_1.json"]
    assert [p.name for p in res.completed] == ["cell_0.json"]
    assert [p.name for p in res.failed] == ["cell_1.json"]
    assert [p.name for p in res.pending] == ["cell_2.json"]


def test_resume_skips_completed(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 3)
    state_path = tmp_path / "s.json"

    def first_run(p: Path) -> None:
        return None

    run_study(specs, run_one=first_run, state_path=state_path)

    seen_again: List[Path] = []

    def second_run(p: Path) -> None:
        seen_again.append(p)

    res = run_study(specs, run_one=second_run, state_path=state_path)
    assert seen_again == []
    assert len(res.skipped) == 3
    assert res.completed == []


def test_resume_retries_failed(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 2)
    state_path = tmp_path / "s.json"

    def first_run(p: Path) -> None:
        if p.name == "cell_1.json":
            raise RuntimeError("boom")

    run_study(specs, run_one=first_run, state_path=state_path)

    seen: List[Path] = []

    def second_run(p: Path) -> None:
        seen.append(p)

    res = run_study(specs, run_one=second_run, state_path=state_path)
    assert [p.name for p in seen] == ["cell_1.json"]
    assert [p.name for p in res.completed] == ["cell_1.json"]
    assert [p.name for p in res.skipped] == ["cell_0.json"]


def test_resume_resets_running_to_pending(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 2)
    state_path = tmp_path / "s.json"

    state = StudyState(
        cells=[
            type("R", (), {})(),
        ]
    )
    from manylatents.runners.sequential_study import CellRecord

    state = StudyState(
        cells=[
            CellRecord(path=str(specs[0]), state=CellState.RUNNING),
            CellRecord(path=str(specs[1]), state=CellState.PENDING),
        ]
    )
    state.save(state_path)

    seen: List[Path] = []

    def run_one(p: Path) -> None:
        seen.append(p)

    res = run_study(specs, run_one=run_one, state_path=state_path)
    assert [p.name for p in seen] == ["cell_0.json", "cell_1.json"]
    assert len(res.completed) == 2


def test_state_file_is_valid_json_between_cells(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 2)
    state_path = tmp_path / "s.json"
    snapshots: List[dict] = []

    def run_one(p: Path) -> None:
        snapshots.append(json.loads(state_path.read_text()))

    run_study(specs, run_one=run_one, state_path=state_path)
    assert len(snapshots) == 2
    for snap in snapshots:
        assert "cells" in snap
        assert all("state" in c and "path" in c for c in snap["cells"])


def test_keyboard_interrupt_propagates_and_marks_failed(tmp_path: Path) -> None:
    specs = _make_specs(tmp_path, 2)
    state_path = tmp_path / "s.json"

    def run_one(p: Path) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_study(specs, run_one=run_one, state_path=state_path)

    state = StudyState.load(state_path)
    assert state.cells[0].state is CellState.FAILED
    assert state.cells[0].error == "KeyboardInterrupt"


def test_resume_appends_new_specs(tmp_path: Path) -> None:
    specs_a = _make_specs(tmp_path, 2)
    state_path = tmp_path / "s.json"

    run_study(specs_a, run_one=lambda p: None, state_path=state_path)

    new_specs = list(specs_a)
    p_extra = tmp_path / "cell_extra.json"
    p_extra.write_text("{}")
    new_specs.append(p_extra)

    seen: List[Path] = []

    def run_one(p: Path) -> None:
        seen.append(p)

    res = run_study(new_specs, run_one=run_one, state_path=state_path)
    assert [p.name for p in seen] == ["cell_extra.json"]
    assert [p.name for p in res.completed] == ["cell_extra.json"]
    assert len(res.skipped) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
