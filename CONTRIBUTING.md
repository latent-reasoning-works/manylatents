# Contributing to manyLatents

Thank you for contributing to manyLatents! This document provides guidelines for adding new components and ensuring they integrate correctly with the framework.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Adding New Metrics](#adding-new-metrics)
- [Adding New Algorithms](#adding-new-algorithms)
- [Adding New Datasets](#adding-new-datasets)
- [Integration Testing via CI](#integration-testing-via-ci)
- [Code Style](#code-style)

---

## Testing Philosophy

All contributions must pass automated tests that run on every pull request. The CI pipeline validates:

1. **Standalone CLI functionality** - Your component works via `python -m manylatents.main`
2. **Integration with existing components** - Your component works with various data/algorithm/metric combinations
3. **Code quality** - Linting, formatting, and type checking pass

**The feedback loop:** When you open a PR, GitHub Actions automatically runs these tests. Failed tests will block merging until resolved.

---

## Adding New Metrics

### Overview

Metrics in manyLatents are organized into three groups based on when they execute:
- **`metrics/dataset/`**: Evaluate original high-dimensional data
- **`metrics/embedding/`**: Evaluate quality of low-dimensional embeddings
- **`metrics/module/`**: Evaluate algorithm internals (graphs, kernels, etc.)

See [`docs/metrics_architecture.md`](docs/metrics_architecture.md) for detailed explanation.

### Step 1: Implement Your Metric

Create your metric function following the required signature:

```python
# manylatents/metrics/your_metric.py
from typing import Optional, Union, Any
import numpy as np

def YourMetric(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    **kwargs  # Your hyperparameters
) -> Union[float, tuple[float, np.ndarray], dict[str, Any]]:
    """
    Compute your metric.

    Args:
        embeddings: The embedding array (or original data for dataset metrics)
        dataset: Dataset object with .data attribute (access to original data)
        module: Algorithm module (access to internals like .graph_, .kernel_)
        **kwargs: Additional hyperparameters

    Returns:
        One of:
        - float: Simple scalar value
        - tuple[float, np.ndarray]: (scalar, per_sample_array) for table logging
        - dict: Structured output for complex metrics
    """
    # Example: return scalar + per-sample
    scalar_value = compute_aggregate(embeddings)
    per_sample_values = compute_per_sample(embeddings)
    return (scalar_value, per_sample_values)
```

**Return Type Guidelines**:
- **Scalar only**: Use `float` for simple summary statistics
- **Scalar + per-sample**: Use `tuple[float, np.ndarray]` to enable wandb table logging
- **Complex output**: Use `dict` for graphs, multiple values, etc.

### Step 2: Create Config

Place the config in the appropriate subdirectory:

```yaml
# manylatents/configs/metrics/embedding/your_metric.yaml
your_metric:
  _target_: manylatents.metrics.your_metric.YourMetric
  _partial_: True
  k: 25
  threshold: 0.5
```

**Choosing the right directory**:
- Does it only need original data? → `metrics/dataset/`
- Does it compare original vs. reduced? → `metrics/embedding/`
- Does it need algorithm internals? → `metrics/module/`

### Step 3: Test Locally

```bash
cd /path/to/manyLatents
source .venv/bin/activate

# Test your metric with fast test components
python -m manylatents.main \
  algorithm=latent/pca \
  data=test_data \
  metrics/embedding=your_metric \
  logger=none
```

**Verify**:
- [ ] Metric computes without errors
- [ ] Scalar value is logged to console
- [ ] If you returned per-sample values, appropriate output is generated
- [ ] Output looks correct

### Step 4: Add to CI Integration Tests (Required)

Add your metric to the CI test matrix to ensure it's validated on every PR. Edit `.github/workflows/build.yml`:

```yaml
# In the integration-tests job, add a new matrix entry:
- name: "pca-yourmetric"
  algorithm: "latent/pca"
  data: "test_data"              # Use fast test data
  metrics: "your_metric"
  timeout: 5
```

**Best practices for CI tests:**
- Use `test_data` and `test_metric` for fastest execution
- Use `swissroll` for synthetic data tests if needed
- Keep timeouts realistic (most tests should be < 10 minutes)

This ensures:
- ✅ Your metric runs on every PR
- ✅ Breaking changes are caught automatically
- ✅ Integration with different algorithms/datasets is validated

### Step 5: Add to Composite Configs (Optional)

If your metric is commonly used, add it to a composite config:

```yaml
# manylatents/configs/metrics/example_suite.yaml
defaults:
  - dataset: null
  - embedding:
    - trustworthiness
    - your_metric  # Add here
    - persistent_homology
  - module: null
  - _self_
```

---

## Adding New Algorithms

### Step 1: Implement Algorithm Module

Inherit from `LatentModule` for dimensionality reduction:

```python
# manylatents/algorithms/latent/your_algorithm.py
from typing import Optional
import torch
from manylatents.algorithms.latent_module_base import LatentModule

class YourAlgorithmModule(LatentModule):
    """Your dimensionality reduction algorithm."""

    def __init__(self, n_components: int = 2, **kwargs):
        super().__init__()
        self.n_components = n_components
        # Initialize your model

    def fit(self, X: torch.Tensor) -> None:
        """Fit the model on training data."""
        # Training logic
        pass

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data to low-dimensional space."""
        # Inference logic
        embeddings = self._compute_embeddings(X)
        return embeddings
```

Or inherit from `LightningModule` for neural networks:
```python
# manylatents/algorithms/lightning/your_network.py
from lightning import LightningModule

class YourNetwork(LightningModule):
    def __init__(self, ...):
        super().__init__()
        # Define architecture

    def training_step(self, batch, batch_idx):
        # Training step
        pass

    def encode(self, x):
        # Extract embeddings
        return embeddings
```

### Step 2: Create Config

```yaml
# manylatents/configs/algorithms/latent/your_algorithm.yaml
_target_: manylatents.algorithms.latent.your_algorithm.YourAlgorithmModule
n_components: 2
learning_rate: 0.001
```

### Step 3: Test Locally

```bash
python -m manylatents.main \
  algorithm=latent/your_algorithm \
  data=test_data \
  metrics=test_metric \
  logger=none
```

**Verify**:
- [ ] Algorithm trains/fits without errors
- [ ] Embeddings are generated
- [ ] Shape is correct (n_samples × n_components)
- [ ] Metrics compute on embeddings

### Step 4: Add to CI Integration Tests (Required)

Add your algorithm to the CI test matrix in `.github/workflows/build.yml`:

```yaml
# In the integration-tests job:
- name: "youralgorithm-test"
  algorithm: "latent/your_algorithm"
  data: "test_data"              # Use fast test data
  metrics: "test_metric"         # Use fast test metric
  timeout: 5
```

**Note:** Use `test_data` and `test_metric` for fast CI runs. This validates your algorithm automatically on every PR.

---

## Adding New Datasets

### Step 1: Implement DataModule

```python
# manylatents/data/your_dataset.py
from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

class YourDataModule(LightningDataModule):
    """Your dataset."""

    def __init__(self, batch_size: int = 32, **kwargs):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """Load/generate data."""
        # Create datasets
        self.train_dataset = YourDataset(split='train')
        self.test_dataset = YourDataset(split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
```

### Step 2: Create Config

```yaml
# manylatents/configs/data/your_dataset.yaml
_target_: manylatents.data.your_dataset.YourDataModule
batch_size: 32
data_path: /path/to/data
```

### Step 3: Test Locally

```bash
python -m manylatents.main \
  data=your_dataset \
  algorithm=latent/pca \
  metrics=test_metric \
  logger=none
```

**Verify**:
- [ ] Data loads without errors
- [ ] Train/test splits are correct
- [ ] Batch shape is correct
- [ ] Algorithm can process the data

### Step 4: Add to CI Integration Tests (Required)

Add your dataset to the CI test matrix in `.github/workflows/build.yml`:

```yaml
# In the integration-tests job:
- name: "pca-yourdataset"
  algorithm: "latent/pca"        # Use simple algorithm
  data: "your_dataset"
  metrics: "test_metric"         # Use fast test metric
  timeout: 8
```

**Note:** Pair with `latent/pca` and `test_metric` for fast validation. This ensures your dataset loads correctly on every PR.

---

## Integration Testing via CI

manyLatents uses GitHub Actions to automatically test every pull request. Understanding this system is key to successful contributions.

### CI Test Jobs

1. **build-and-test** (runs first)
   - Installs dependencies
   - Runs basic CLI smoke test
   - Fast feedback (~2-3 minutes)

2. **integration-tests** (runs after build-and-test passes)
   - Matrix of algorithm/data/metric combinations
   - Each combination runs independently
   - Tests your component with various configurations
   - Longer runtime (~10-25 minutes total)

### Adding Your Component to CI

When you add a new algorithm, dataset, or metric, **you must add it to the integration test matrix**:

**Edit `.github/workflows/build.yml`:**

```yaml
integration-tests:
  strategy:
    matrix:
      test-config:
        # ... existing tests ...

        # Add your new test:
        - name: "descriptive-test-name"
          algorithm: "latent/pca"       # Use fast test components
          data: "test_data"             # test_data or swissroll for speed
          metrics: "test_metric"        # test_metric for speed
          timeout: 5                    # Keep realistic and short
```

**Recommended test component combinations for speed:**
- **Fastest**: `latent/pca` + `test_data` + `test_metric` (~2-5 min)
- **Synthetic**: `latent/pca` + `swissroll` + `test_metric` (~5-8 min)
- **Neural net**: `lightning/ae_reconstruction` + `test_data` + `test_metric` + `trainer.fast_dev_run=true` (~5-10 min)

**TODO:** Add a Lightning module test to the CI workflow matrix (currently missing). Should include `trainer.fast_dev_run=true` to only run a few batches for fast validation.

### PR Validation Flow

1. **Open PR** → CI automatically runs
2. **View test results** → Check the "Actions" tab on GitHub
3. **Fix failures** → Push new commits to trigger re-run
4. **All green** → PR is ready for review

### Local Pre-submission Checklist

Before opening your PR, verify locally:

- [ ] Component works in isolation (`python -m manylatents.main ...`)
- [ ] No errors in console output
- [ ] Output files created in `outputs/`
- [ ] Correct shapes and values for outputs
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Added CI test matrix entry in `.github/workflows/build.yml`
- [ ] Docstrings added to new functions/classes
- [ ] Config files documented with comments

---

## Code Style

- **Follow PEP 8** for Python code
- **Use type hints** for function signatures
- **Add docstrings** for all public functions and classes
- **Keep functions focused** - single responsibility principle
- **Use meaningful names** for variables and functions

Example:
```python
def compute_trustworthiness(
    embeddings: np.ndarray,
    original_data: np.ndarray,
    n_neighbors: int = 5
) -> float:
    """
    Compute trustworthiness metric.

    Args:
        embeddings: Low-dimensional embeddings
        original_data: High-dimensional original data
        n_neighbors: Number of neighbors to consider

    Returns:
        Trustworthiness score between 0 and 1
    """
    # Implementation
    pass
```

---

## Questions?

- **Architecture questions**: See `docs/metrics_architecture.md`
- **API usage**: See `docs/api_usage.md`
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions

Thank you for contributing to manyLatents!
