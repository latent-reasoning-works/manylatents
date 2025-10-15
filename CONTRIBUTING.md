# Contributing to manyLatents

Thank you for contributing to manyLatents! This document provides guidelines for adding new components and ensuring they work correctly both in the CLI and when orchestrated through manyAgents.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Adding New Metrics](#adding-new-metrics)
- [Adding New Algorithms](#adding-new-algorithms)
- [Adding New Datasets](#adding-new-datasets)
- [Code Style](#code-style)

---

## Testing Philosophy

manyLatents is designed to work both:
1. **Standalone** via the CLI (`python -m manylatents.main`)
2. **Orchestrated** through manyAgents (for multi-tool workflows)

**When you add a new component, you must test both paths.**

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

### Step 3: Test Standalone (manyLatents CLI)

```bash
cd /path/to/manyLatents
source .venv/bin/activate

# Test your metric in isolation
python -m manylatents.main \
  experiment=single_algorithm \
  metrics/embedding=your_metric \
  debug=true
```

**Verify**:
- [ ] Metric computes without errors
- [ ] Scalar value is logged to console
- [ ] If you returned per-sample values, wandb table is created
- [ ] Output looks correct

### Step 4: Test Through manyAgents (Integration Test)

**REQUIRED**: Ensure your metric works when called through orchestration.

```bash
cd /path/to/manyAgents
source .venv/bin/activate
uv sync  # Get latest manylatents with your changes

# Run the integration test
python -m manyagents.main experiment=manylatents_pipeline_with_metrics
```

This test validates:
- ✅ Config overrides work through the adapter
- ✅ Metric computes in multi-step pipelines
- ✅ Results are properly returned to orchestrator
- ✅ Wandb logging works end-to-end
- ✅ In-memory data passing between steps

**If this fails, your metric is not ready for merge.**

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

### Step 3: Test Standalone

```bash
python -m manylatents.main \
  algorithms/latent=your_algorithm \
  data=swissroll \
  metrics=test_metric \
  debug=true
```

**Verify**:
- [ ] Algorithm trains/fits without errors
- [ ] Embeddings are generated
- [ ] Shape is correct (n_samples × n_components)
- [ ] Metrics compute on embeddings

### Step 4: Test Through manyAgents

```bash
cd /path/to/manyAgents

# Create a test workflow or override existing
python -m manyagents.main \
  experiment=manylatents_single_algorithm \
  workflow.steps.0.config.algorithms.latent._target_=manylatents.algorithms.latent.your_algorithm.YourAlgorithmModule
```

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

### Step 3: Test Standalone

```bash
python -m manylatents.main \
  data=your_dataset \
  algorithms/latent=pca \
  metrics=test_metric \
  debug=true
```

**Verify**:
- [ ] Data loads without errors
- [ ] Train/test splits are correct
- [ ] Batch shape is correct
- [ ] Algorithm can process the data

### Step 4: Test Through manyAgents

```bash
cd /path/to/manyAgents
python -m manyagents.main \
  experiment=manylatents_single_algorithm \
  workflow.steps.0.config.data=your_dataset
```

---

## Integration Testing Checklist

Before submitting a PR with new components:

### 1. Component Works in Isolation
```bash
cd /path/to/manyLatents
python -m manylatents.main <your-test-command>
```

### 2. Component Works Through Orchestration
```bash
cd /path/to/manyAgents
uv lock --upgrade-package manylatents  # Get latest with your changes
uv sync
python -m manyagents.main experiment=manylatents_pipeline_with_metrics
```

### 3. Check Outputs
- [ ] No errors in console
- [ ] Output files created in `outputs/`
- [ ] Wandb logs populated correctly (if not in debug mode)
- [ ] Metrics/embeddings have correct values and shapes

### 4. Documentation
- [ ] Docstrings added to new functions/classes
- [ ] Config files documented with comments
- [ ] README updated if adding major feature

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
