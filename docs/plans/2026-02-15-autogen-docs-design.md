# Auto-Generated Documentation Design

## Goal

Replace hand-maintained tables in docs with auto-generated tables populated from the codebase (Hydra configs, `@register_metric` registry, docstrings). Host on GitHub Pages via MkDocs Material.

## Architecture

**Hybrid per-page layout**: Hand-written context/explanation stays at the top of each page. Auto-generated tables are injected below via Jinja2 macro calls. One `mkdocs build` does everything.

## Tooling

| tool | role |
|---|---|
| `mkdocs-material` | theme (already installed) |
| `mkdocs-macros-plugin` | Jinja2 macros in markdown |
| `mkdocstrings[python]` | API reference from docstrings (follow-up) |
| GitHub Actions | build verification + `mkdocs gh-deploy` |

## Macro Hook

`docs/macros.py` — registered as `module_name` in `mkdocs.yml`.

Four macros:

| macro | data source | output |
|---|---|---|
| `algorithm_table(type)` | `configs/algorithms/{type}/*.yaml` | algorithm, config override, key params |
| `metrics_table(context)` | `get_metric_registry()` + `configs/metrics/{context}/*.yaml` | metric, config, default params, description |
| `data_table()` | `configs/data/*.yaml` | dataset, config, key params |
| `sampling_table()` | `configs/metrics/sampling/*.yaml` | strategy, config, method |

The metric registry is the richest source: it provides description, aliases, and default_params via the `@register_metric` decorator. Configs provide the Hydra override name.

## Page Structure

Existing hand-written pages stay. Macro calls replace manual tables:

```markdown
## embedding metrics

Compare high-dimensional input to low-dimensional output.

{{ metrics_table("embedding") }}

Config pattern: `metrics/embedding=<name>`
```

Applied to: `algorithms.md`, `metrics.md`, `evaluation.md` (sampling), new `data.md`.

Per-function API reference pages via mkdocstrings are a follow-up.

## Nav

```yaml
nav:
  - Home: index.md
  - Algorithms: algorithms.md
  - Data: data.md
  - Metrics: metrics.md
  - Evaluation: evaluation.md
  - Cache Protocol: cache.md
  - Callbacks: callbacks.md
  - Extensions: extensions.md
  - API: api_usage.md
  - Probing: probing.md
  - Testing: testing.md
```

## CI

Three jobs:

1. **Docs build** (every push/PR): `uv run mkdocs build --strict` — fails on broken links, macro errors, missing references.

2. **Docs completeness** (every push/PR): `scripts/check_docs_coverage.py` asserts:
   - Every config in `configs/metrics/` has a matching `@register_metric` entry
   - Every registered metric has a non-empty docstring
   - Every config in `configs/algorithms/` resolves to a valid `_target_`

3. **Deploy** (push to main only, after 1+2 pass): `uv run mkdocs gh-deploy --force`

## README

Slim down to landing page. Replace tables with one-line summaries + links to docs site:

```markdown
## algorithms

> 12 algorithms -- 8 latent modules, 4 lightning modules

[Full reference ->](https://latent-reasoning-works.github.io/manylatents/algorithms/)
```

Keep: ASCII header, quickstart, architecture diagram, cache protocol snippet.

## Out of Scope (follow-ups)

- Per-function API reference pages via mkdocstrings
- Auto-generated mkdocstrings directives from registry
- Cross-repo LRW docs unification
- Search integration
