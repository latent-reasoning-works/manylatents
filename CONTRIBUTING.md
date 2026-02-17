# Contributing to manyLatents

Guidelines for adding new components and ensuring they integrate correctly with the framework.

## Testing Philosophy

All contributions must pass automated tests on every pull request. CI validates:

1. **Unit tests** — `pytest tests/` runs the full test suite
2. **CLI smoke tests** — default LatentModule and LightningModule paths work
3. **Component discovery** — if your PR touches algorithms, metrics, or data configs, CI auto-discovers all configs in that group and runs each one end-to-end
4. **Docs build** — `mkdocs build --strict` passes

### How CI detects what to test

CI uses **path-based filtering**. If your PR modifies files under a component directory, the corresponding discovery test runs automatically:

| Files changed | CI runs |
|---|---|
| `manylatents/algorithms/latent/**` or `configs/algorithms/latent/**` | Discovers and smoke-tests **every** LatentModule config |
| `manylatents/algorithms/lightning/**` or `configs/algorithms/lightning/**` | Discovers and smoke-tests **every** LightningModule config |
| `manylatents/metrics/**` or `configs/metrics/**` | Discovers and smoke-tests **every** metric config |

This means: if you add a new config YAML and its `_target_` doesn't resolve, or the algorithm crashes on synthetic data, CI will catch it.

---

## Adding New Metrics

Every new metric follows 4 steps: **wrapper → register → config → smoke test**.

### Step 1: Write the Wrapper

Follow the `Metric` protocol — `embeddings` first, then `dataset`, `module`, your params, and `cache`:

```python
# manylatents/metrics/your_metric.py
from typing import Optional
import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn


@register_metric(
    aliases=["your_metric"],
    default_params={"k": 25},
    description="Short description of what this metric measures",
)
def YourMetric(
    embeddings: np.ndarray,
    dataset=None,
    module=None,
    k: int = 25,
    cache: Optional[dict] = None,
) -> float:
    # Use compute_knn with cache for shared kNN computation
    dists, indices = compute_knn(embeddings, k=k, cache=cache)
    score = ...  # your computation
    return float(score)
```

**Return types:** `float`, `tuple[float, np.ndarray]` (scalar + per-sample), or `dict[str, Any]` (structured).

**Evaluation context** determines the config directory:

- Only needs original data? → `metrics/dataset/`
- Compares original vs. reduced? → `metrics/embedding/`
- Needs algorithm internals (graph, kernel)? → `metrics/module/`

### Step 2: Register It

The `@register_metric` decorator (shown above) adds your metric to the dynamic registry with aliases, default params, and a description. This powers `list_metrics()`, auto-generated docs tables, and programmatic discovery.

Import your metric in `manylatents/metrics/__init__.py` so the decorator fires at import time.

### Step 3: Create the Config

```yaml
# manylatents/configs/metrics/embedding/your_metric.yaml
your_metric:
  _target_: manylatents.metrics.your_metric.YourMetric
  _partial_: True
  k: 25
```

Configs are **nested under the metric name** with `_partial_: True` so Hydra binds the params at config time and the engine calls it with `embeddings`, `dataset`, `module`, and `cache` at runtime.

### Step 4: E2E Smoke Test

```bash
# Verify your metric runs end-to-end
manylatents algorithms/latent=pca data=swissroll \
  metrics/embedding=your_metric logger=none
```

CI will auto-discover your new config and test it if `manylatents/metrics/` or `configs/metrics/` files are changed.

---

## Adding New Algorithms

### LatentModule (non-neural)

For algorithms without a training loop — fit/transform pattern:

```python
# manylatents/algorithms/latent/your_algorithm.py
import numpy as np
from manylatents.algorithms.latent.latent_module_base import LatentModule


class YourAlgorithmModule(LatentModule):
    """Your dimensionality reduction algorithm."""

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__()
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> None:
        """Fit the model on training data."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to low-dimensional space."""
        return self._compute_embeddings(X)
```

### LightningModule (neural)

For algorithms that train with backprop:

```python
# manylatents/algorithms/lightning/your_network.py
from lightning import LightningModule


class YourNetwork(LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule", "network", "loss"])

    def training_step(self, batch, batch_idx):
        pass

    def encode(self, x):
        return embeddings

    def configure_optimizers(self):
        return ...
```

### Create Config

```yaml
# manylatents/configs/algorithms/latent/your_algorithm.yaml
_target_: manylatents.algorithms.latent.your_algorithm.YourAlgorithmModule
n_components: 2
```

### Test Locally

```bash
# LatentModule
manylatents algorithms/latent=your_algorithm data=swissroll \
  metrics=noop logger=none

# LightningModule
manylatents algorithms/lightning=your_network data=swissroll \
  trainer.fast_dev_run=true metrics=noop logger=none
```

CI will auto-discover your new config and test it when you open a PR.

---

## Adding New Datasets

```python
# manylatents/data/your_dataset.py
from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class YourDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ...
        self.test_dataset = ...

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
```

```yaml
# manylatents/configs/data/your_dataset.yaml
_target_: manylatents.data.your_dataset.YourDataModule
batch_size: 32
```

```bash
manylatents data=your_dataset algorithms/latent=pca \
  metrics=noop logger=none
```

---

## CI Pipeline

Three jobs run on every PR:

**test** (Python 3.11 + 3.12 matrix):
- `pytest tests/` — full unit test suite
- CLI smoke test: LatentModule path (`experiment=single_algorithm`)
- CLI smoke test: LightningModule path (`algorithms/lightning=ae_reconstruction`, `trainer.fast_dev_run=true`)
- If `algorithms/latent/` changed → discovers and tests **all** LatentModule configs
- If `algorithms/lightning/` changed → discovers and tests **all** LightningModule configs
- If `metrics/` changed → discovers and tests **all** metric configs

**docs:**
- `scripts/check_docs_coverage.py` — verifies all `_target_` paths in configs are importable
- `mkdocs build --strict` — verifies docs site builds cleanly

**publish** (tags only):
- Builds sdist + wheel, publishes to PyPI via trusted publishing

### Local Pre-submission Checklist

```bash
# Run unit tests
pytest tests/ -x -q

# Smoke test your component
manylatents algorithms/latent=your_algo data=swissroll metrics=noop logger=none

# Docs build (optional)
mkdocs build --strict
```

---

## Optional Dependencies

Some features require optional extras. If your contribution uses an optional dependency:

1. Add it to the appropriate `[project.optional-dependencies]` group in `pyproject.toml`
2. Use lazy imports (`try/except ImportError`) so the core package doesn't break without it
3. Add `pytest.importorskip()` or `@pytest.mark.skipif` to tests that need the dep

Current extras: `tracking` (wandb), `hf` (transformers), `dynamics` (torchdiffeq/torchsde), `transport` (POT), `topology` (ripser), `cluster` (submitit), `torchdr`, `docs`.

---

## Questions?

- [Metrics reference](https://latent-reasoning-works.github.io/manylatents/metrics/)
- [API usage](https://latent-reasoning-works.github.io/manylatents/api_usage/)
- [Cache protocol](https://latent-reasoning-works.github.io/manylatents/cache/)
- Issues: open a [GitHub issue](https://github.com/latent-reasoning-works/manylatents/issues)
