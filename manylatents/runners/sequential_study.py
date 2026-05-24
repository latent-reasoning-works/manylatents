"""Run a sequence of cell specs in one Python process with resumable state.

Designed for multi-cell distillation studies where per-cell SLURM overhead
dominates. One sbatch wraps a Python loop; the loop persists per-cell state to
JSON so a kill mid-cell resumes cleanly. Per-cell Lightning checkpointing is
the caller's responsibility.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Sequence

__all__ = [
    "CellState",
    "CellRecord",
    "StudyResult",
    "StudyState",
    "run_study",
]


class CellState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CellRecord:
    path: str
    state: CellState = CellState.PENDING
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class StudyResult:
    completed: List[Path] = field(default_factory=list)
    failed: List[Path] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)
    pending: List[Path] = field(default_factory=list)


@dataclass
class StudyState:
    cells: List[CellRecord]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"cells": [_record_to_dict(c) for c in self.cells]}
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "StudyState":
        with open(path, "r") as f:
            payload = json.load(f)
        return cls(cells=[_record_from_dict(d) for d in payload["cells"]])


def _record_to_dict(rec: CellRecord) -> dict:
    d = asdict(rec)
    d["state"] = rec.state.value
    return d


def _record_from_dict(d: dict) -> CellRecord:
    return CellRecord(
        path=d["path"],
        state=CellState(d["state"]),
        error=d.get("error"),
        started_at=d.get("started_at"),
        completed_at=d.get("completed_at"),
    )


def _merge_specs_into_state(
    specs: Sequence[Path], state: StudyState
) -> StudyState:
    by_path = {c.path: c for c in state.cells}
    out: List[CellRecord] = []
    for spec in specs:
        spath = str(spec)
        if spath in by_path:
            rec = by_path[spath]
            if rec.state is CellState.RUNNING:
                rec.state = CellState.PENDING
                rec.started_at = None
            out.append(rec)
        else:
            out.append(CellRecord(path=spath))
    return StudyState(cells=out)


def run_study(
    specs: Sequence[Path],
    *,
    run_one: Callable[[Path], Any],
    state_path: Path,
    on_failure: Literal["skip", "halt"] = "skip",
) -> StudyResult:
    """Run each spec in ``specs`` sequentially, recording state to ``state_path``.

    Resume contract:
      * COMPLETED cells are skipped.
      * FAILED, RUNNING, and PENDING cells are retried (RUNNING is reset to
        PENDING on entry — handles a kill mid-cell).
      * New paths in ``specs`` not in the prior state file are appended.

    Failure policy:
      * ``on_failure="skip"`` records the failure and continues.
      * ``on_failure="halt"`` records the failure and stops.
      * ``KeyboardInterrupt`` always propagates after marking the active cell
        FAILED and persisting state.
    """
    state_path = Path(state_path)
    if state_path.exists():
        state = _merge_specs_into_state(specs, StudyState.load(state_path))
    else:
        state = StudyState(cells=[CellRecord(path=str(s)) for s in specs])

    state.save(state_path)
    result = StudyResult()

    for rec in state.cells:
        spec_path = Path(rec.path)
        if rec.state is CellState.COMPLETED:
            result.skipped.append(spec_path)
            continue

        rec.state = CellState.RUNNING
        rec.started_at = time.time()
        rec.error = None
        rec.completed_at = None
        state.save(state_path)

        try:
            run_one(spec_path)
        except KeyboardInterrupt:
            rec.state = CellState.FAILED
            rec.error = "KeyboardInterrupt"
            rec.completed_at = time.time()
            state.save(state_path)
            raise
        except Exception as exc:
            rec.state = CellState.FAILED
            rec.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rec.completed_at = time.time()
            state.save(state_path)
            result.failed.append(spec_path)
            if on_failure == "halt":
                break
            continue

        rec.state = CellState.COMPLETED
        rec.completed_at = time.time()
        state.save(state_path)
        result.completed.append(spec_path)

    for rec in state.cells:
        if rec.state is CellState.PENDING:
            result.pending.append(Path(rec.path))

    return result
