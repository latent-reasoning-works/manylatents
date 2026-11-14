# AGENTS.md

Instructions for coding agents working in this repository.

The contributor rules are in three files. Read them before changing anything:

- [`CLAUDE.md`](CLAUDE.md): working rules, entry points, how to add a component, gotchas.
  It applies to every agent, not only to Claude.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): codebase map, data flow, architectural invariants.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the steps for adding a metric or a module.

If this file and `CLAUDE.md` disagree, `CLAUDE.md` is right.

## The rules that matter most

- Run everything with `uv run`, never bare `python`.
- Before pushing, run:
  `uv run pytest tests/ -x -q && uv run pytest manylatents/callbacks/tests/ -x -q`
- Keep changes minimal and scoped to the task. Ask before doing extra work.
- Check `git log` and merged pull requests for an existing implementation before building
  a new one.
- Use the manylatents API (`from manylatents.api import run`) and call `run()` directly. Do
  not wrap it in helper functions or pipeline classes.
- Optional dependencies are imported inside methods. A module file must import cleanly
  without them.
- Metrics that need kNN use `compute_knn()` from `manylatents/utils/metrics.py`, with the
  shared `cache`.
- Every new LatentModule or metric needs its Hydra config. CI smoke-tests every config.
- This is a public repository. Do not push experiment configs, sweeps, analysis scripts or
  data-preparation scripts.
- Do not open a pull request, push to `main`, or tag a release unless asked.
