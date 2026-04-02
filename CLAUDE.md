# CLAUDE.md

Unified dimensionality reduction and neural network analysis. PyTorch Lightning + Hydra + uv.

**See [ARCHITECTURE.md](ARCHITECTURE.md) for the codebase map, data flow, and architectural invariants.**

## Before Starting Work

- **Check for existing implementations** before building anything new. Run `git log --oneline main | head -30` and `gh pr list --state merged --limit 10` to avoid reimplementing features that already exist under different naming.
- **Always use manylatents APIs** (e.g. `from manylatents.api import run`) for experiments and traces — never substitute with raw sklearn/numpy equivalents unless explicitly told to.
- **Prefer minimal, scoped changes.** Do not proactively expand scope (adding config files, fixing docs, cleaning orphan configs) unless explicitly asked. Ask before doing extra work.

## Using the Python API

**Call `run()` directly with parameters. Do not wrap it.** The API is designed for functional composition — pass data, algorithms, and metrics as arguments. Do not create helper functions, wrapper classes, or "pipeline" abstractions around it.

```python
# CORRECT — direct functional call
from manylatents.api import run

result = run(
    data="swissroll",
    algorithms={"latent": "pca"},
    metrics={"trustworthiness": {
        "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
        "_partial_": True, "n_neighbors": 5,
    }},
)
result["embeddings"]  # (n, d) ndarray
result["scores"]      # {"trustworthiness": 0.95}
```

```python
# WRONG — unnecessary wrapper
def run_pca_analysis(data_path, n_components=2):
    """Don't do this."""
    module = PCAModule(n_components=n_components)
    data = load_data(data_path)
    module.fit(data)
    return module.transform(data)
```

When writing analysis scripts, parameterize through `run()` arguments or Hydra config overrides, not through custom wrapper functions. If you need to vary parameters, use a loop over `run()` calls or a Hydra multirun.

## CLI Entry Points

```bash
# CLI — primary interface
uv run python -m manylatents.main algorithms/latent=pca data=swissroll metrics=trustworthiness

# LightningModule path
uv run python -m manylatents.main algorithms/lightning=ae_reconstruction data=swissroll trainer.fast_dev_run=true

# Multirun sweep
uv run python -m manylatents.main --multirun algorithms/latent=umap,phate,tsne data=swissroll metrics=trustworthiness

# SLURM submission
uv run python -m manylatents.main -m cluster=mila resources=gpu algorithms/latent=umap data=swissroll
```

## What belongs here

This is a **public** repo. Only core infrastructure goes here:
- New LatentModules, metrics, callbacks, data modules
- Bug fixes, performance improvements, refactoring
- Generic config groups (algorithms, data, metrics, callbacks, cluster, resources)

Hydra instantiation configs (algorithm, data, metric, callback YAMLs) belong here — they're part of the core and CI depends on them.

**Do NOT push** experiment configs (sweep definitions), analysis scripts, data prep scripts, or project-specific sweeps. Those belong in the downstream repo that consumes manylatents (expaper repos, practitioner repos, etc.) — each has its own `experiments/configs/manylatents/experiment/` directory and a local manylatents pin.

## Releases

When tagging a release:
1. **Bump `version` in `pyproject.toml` first** — PyPI rejects uploads if the version already exists.
2. Run `uv run pytest tests/ -x -q` to verify.
3. Commit the version bump, push to main, then create the tag/release.
4. Watch the Publish to PyPI workflow with `gh run watch`.

## Pre-push checklist

**CI must pass before pushing to main.** Run:

```bash
uv run pytest tests/ -x -q && uv run pytest manylatents/callbacks/tests/ -x -q
```

If `gh` CLI is available, run CI locally **before pushing** to catch runner-specific failures early:

```bash
gh act -W .github/workflows/ci.yml   # run CI workflow locally via act
```

If `act` is not installed, at minimum verify checks immediately after push and fix before merging:

```bash
gh run watch   # watch the CI run after pushing
```

If CI fails after pushing, fix immediately — do not leave main broken.

## Running Experiments

**Always run experiment submissions as background tasks.** SLURM submissions and multirun sweeps can take time to dispatch — use `run_in_background: true` on the Bash tool so the conversation isn't blocked waiting. This applies to any `uv run python -m manylatents.main` invocation that submits to a cluster or runs a sweep.

## Config Discovery

**Don't hardcode config names** — discover them:

```bash
ls manylatents/configs/algorithms/latent/   # available LatentModule configs
ls manylatents/configs/algorithms/lightning/ # available LightningModule configs
ls manylatents/configs/data/                # datasets
ls manylatents/configs/metrics/             # all metrics (flat, each has at: field)
ls manylatents/configs/callbacks/embedding/ # callbacks
ls manylatents/configs/cluster/             # cluster profiles
```

For registered metrics: `uv run python -c "from manylatents.metrics import list_metrics; print(list_metrics())"`

## Adding New Components

**New metric**: wrapper function → `@register_metric` decorator → config YAML in `configs/metrics/<name>.yaml` (flat, with `at:` field) → import in `__init__.py` → CI smoke test.
See `CONTRIBUTING.md` for the full 4-step pipeline.

**New LatentModule** — there are exactly 4 files to touch:

1. **Module**: `manylatents/algorithms/latent/<name>.py`
   - Subclass `LatentModule`
   - Subclasses accept `random_state` as a constructor param (familiar to sklearn users), then pass it to the base class as `init_seed`: `super().__init__(n_components=n_components, init_seed=random_state, neighborhood_size=neighborhood_size, backend=backend, device=device, **kwargs)`
   - `neighborhood_size` overrides any module-specific neighbor count (e.g. `self.n_neighbors = neighborhood_size if neighborhood_size is not None else n_neighbors`)
   - `fit(x: Tensor)` — fit on data, set `self._is_fitted = True`
   - `transform(x: Tensor) -> Tensor` — return embeddings
   - Use shared infra: `compute_knn()` from `utils/metrics.py` (FAISS-GPU cache), not sklearn directly
   - Keep third-party imports **lazy** (inside methods) if the dep is optional — the module file must import cleanly without the optional dep installed

2. **Export**: `manylatents/algorithms/latent/__init__.py`
   - Add `from .<name> import <Name>Module` and add to `__all__`
   - No try/except guard — lazy imports in the module file handle missing optional deps

3. **Hydra config**: `manylatents/configs/algorithms/latent/<name>.yaml`
   - `_target_: manylatents.algorithms.latent.<name>.<Name>Module`
   - Use `random_state: ${seed}` and `neighborhood_size: ${neighborhood_size}`
   - Set `backend: null` and `device: null` only for TorchDR-capable modules; omit for others

4. **Test**: `tests/test_<name>.py`
   - Use `pytest.importorskip("<optional_dep>")` at module level for optional deps
   - Test `fit_transform()` returns correct shape
   - Test the module `isinstance(m, LatentModule)`
   - Test determinism (same seed → same output, use `np.allclose` for numerical methods)

**New LightningModule**: subclass → implement `setup()` + `training_step()` + `encode()` + `configure_optimizers()` → config YAML.

## Gotchas

- **`uv run`, not `python`** — always prefix with `uv run` or activate the venv.
- **Hydra null** — CLI doesn't support `callbacks=null`. Use `logger=none` or explicit null configs.
- **API metrics need `_target_`** — empty dicts `{}` are silently skipped by `flatten_and_unroll_metrics()` in `utils/metrics.py`.
- **LightningModule unit tests** — must call `model.setup()` if not using `trainer.fit()`.
- **`save_hyperparameters`** — always `ignore=["datamodule", "network", "loss"]`.
- **Loss functions** — use project's `MSELoss` (`outputs, targets, **kwargs`), not `torch.nn.MSELoss`.
- **`neighborhood_size` is the unified neighbor parameter** — always use `neighborhood_size=k` when constructing any LatentModule, NOT method-specific params (`n_neighbors`, `knn`, `perplexity`). The base class routes `neighborhood_size` to each method's internal param. Using `n_neighbors=k` on TSNEModule silently goes to `**kwargs` and is ignored (perplexity stays at default 30).
- **Metric `at:` contexts** — `embedding` passes the low-dim embeddings, `dataset` passes the original high-dim input data (as the `embeddings` arg — naming is confusing but intentional), `module` makes the fitted module available.

---

*Last updated: 2026-04-01*
