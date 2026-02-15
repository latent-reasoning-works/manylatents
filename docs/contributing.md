# Contributing to manyLatents

Guidelines for adding new components and ensuring they integrate correctly with the framework.

## Testing Philosophy

All contributions must pass automated tests on every pull request. CI validates:

1. **CLI functionality** — your component works via `uv run python -m manylatents.main`
2. **Integration** — works with existing data/algorithm/metric combinations
3. **Docs build** — `mkdocs build --strict` passes, docs coverage checker finds no stale targets

---

## Adding New Metrics

### Overview

Metrics are organized into three evaluation contexts:

- **`metrics/dataset/`** — properties of the original high-dimensional data
- **`metrics/embedding/`** — quality of low-dimensional embeddings
- **`metrics/module/`** — algorithm internals (graphs, kernels, spectra)

See the [Metrics docs](https://latent-reasoning-works.github.io/manylatents/metrics/) for the full reference.

### Step 1: Implement Your Metric

Follow the `Metric` protocol — `embeddings` first, then `dataset`, `module`, your params, and `cache`:

```python
# manylatents/metrics/your_metric.py
from typing import Optional, Any
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
    """Compute your metric.

    Args:
        embeddings: Low-dimensional embedding array (n, d).
        dataset: Dataset object with .data attribute.
        module: Fitted algorithm module.
        k: Number of neighbors.
        cache: Shared kNN/eigenvalue cache dict.

    Returns:
        Scalar score.
    """
    # Use compute_knn with cache for shared kNN computation
    dists, indices = compute_knn(embeddings, k=k, cache=cache)

    score = ...  # your computation
    return float(score)
```

**Return types:**

| Type | Use case |
|------|----------|
| `float` | Simple scalar |
| `tuple[float, np.ndarray]` | Scalar + per-sample values (enables wandb table logging) |
| `dict[str, Any]` | Structured output (multiple values, arrays) |

### Step 2: Create Config

```yaml
# manylatents/configs/metrics/embedding/your_metric.yaml
_target_: manylatents.metrics.your_metric.YourMetric
_partial_: true
k: 25
```

Note: configs are **flat** — `_target_` at the top level, not nested under a key.

**Choosing the right directory:**

- Only needs original data? → `metrics/dataset/`
- Compares original vs. reduced? → `metrics/embedding/`
- Needs algorithm internals? → `metrics/module/`

### Step 3: Test Locally

```bash
uv run python -m manylatents.main \
  algorithms/latent=pca data=test_data \
  metrics/embedding=your_metric logger=none
```

### Step 4: Verify CI Passes

```bash
# smoke test (what CI runs)
uv run python -m manylatents.main \
  experiment=single_algorithm metrics=test_metric \
  callbacks/embedding=minimal logger=none

# docs coverage (checks _target_ paths are importable)
uv run python scripts/check_docs_coverage.py
```

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
uv run python -m manylatents.main \
  algorithms/latent=your_algorithm data=test_data \
  metrics=test_metric logger=none

# LightningModule
uv run python -m manylatents.main \
  algorithms/lightning=your_network data=swissroll \
  trainer.fast_dev_run=true logger=none
```

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
uv run python -m manylatents.main \
  data=your_dataset algorithms/latent=pca \
  metrics=test_metric logger=none
```

---

## CI Pipeline

Two jobs run on every PR:

**build-and-test:**
- Installs dependencies with uv
- CLI smoke test with `experiment=single_algorithm`
- Smoke tests all LatentModule algorithms (if `algorithms/latent/` files changed)

**docs:**
- Runs `scripts/check_docs_coverage.py` — verifies all `_target_` paths are importable
- Runs `mkdocs build --strict` — verifies docs site builds cleanly

### Local Pre-submission Checklist

- [ ] Component works in isolation
- [ ] CI smoke test passes:
  ```bash
  uv run python -m manylatents.main \
    experiment=single_algorithm metrics=test_metric \
    callbacks/embedding=minimal logger=none
  ```
- [ ] Docs build: `uv run mkdocs build --strict`
- [ ] Tests pass: `uv run pytest tests/ -v`

---

## Questions?

- [Metrics reference](https://latent-reasoning-works.github.io/manylatents/metrics/)
- [API usage](https://latent-reasoning-works.github.io/manylatents/api_usage/)
- [Cache protocol](https://latent-reasoning-works.github.io/manylatents/cache/)
- Issues: open a [GitHub issue](https://github.com/latent-reasoning-works/manylatents/issues)
