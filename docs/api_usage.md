# API Usage Guide

ManyLatents has two APIs: a **direct Python API** for fast iteration, and a **config-driven API** (Hydra) for reproducible sweeps and SLURM submission.

## Quick Start (Direct API)

```python
import numpy as np
from manylatents.algorithms.latent import UMAPModule, PHATEModule
from manylatents import evaluate_metrics

# Generate or load data
X = np.random.randn(500, 50).astype(np.float32)

# Fit a DR method — accepts ndarray directly
mod = UMAPModule(n_components=2, n_neighbors=15)
emb = mod.fit_transform(X)  # ndarray in, ndarray out

# Evaluate metrics by name
scores = evaluate_metrics(emb, metrics=["FractalDimension", "Anisotropy"])

# Module-context metrics (need the fitted module)
scores = evaluate_metrics(emb, module=mod, metrics=["k_eff", "trustworthiness"])
```

## Shared Cache

When evaluating multiple methods on the same data, pass a shared `cache` dict to avoid redundant kNN computation:

```python
cache = {}

mod1 = UMAPModule(n_components=2, n_neighbors=15)
emb1 = mod1.fit_transform(X)
scores1 = evaluate_metrics(emb1, module=mod1, metrics=["k_eff"], cache=cache)

mod2 = PHATEModule(n_components=2, knn=15, n_landmark=None)
emb2 = mod2.fit_transform(X)
scores2 = evaluate_metrics(emb2, module=mod2, metrics=["k_eff"], cache=cache)
# Input-space kNN not recomputed
```

## Config-Driven API (Hydra)

For reproducible sweeps and SLURM submission, use the Hydra-based `api.run()`:

```python
from manylatents.api import run

# Single algorithm run
result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 10}}
)

embeddings = result['embeddings']  # numpy array
scores = result['scores']          # dict of metrics
```

### Chained Workflows

Chain multiple algorithms by passing the output of one as input to another:

```python
from manylatents.api import run

# Step 1: Initial dimensionality reduction
result1 = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 50}}
)

# Step 2: Chain to another algorithm
result2 = run(
    input_data=result1['embeddings'],
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.umap.UMAPModule', 'n_components': 2}}
)

final_embeddings = result2['embeddings']
```

### `run(input_data=None, **overrides)`

Execute a dimensionality reduction algorithm.

**Parameters:**

- `input_data` (np.ndarray, optional): In-memory data array of shape `(n_samples, n_features)`. If provided, this data is used instead of loading from a dataset.
- `**overrides`: Configuration overrides (e.g., `data='swissroll'`, `algorithms={...}`)

**Returns:**

Dictionary with keys:
- `embeddings`: Computed embeddings (numpy array)
- `label`: Labels from dataset (if available)
- `metadata`: Run metadata dictionary
- `scores`: Evaluation metrics (if enabled)

## Available Algorithms

### Dimensionality Reduction

- **PCA**: `manylatents.algorithms.latent.pca.PCAModule`
- **t-SNE**: `manylatents.algorithms.latent.tsne.TSNEModule`
- **UMAP**: `manylatents.algorithms.latent.umap.UMAPModule`
- **PHATE**: `manylatents.algorithms.latent.phate.PHATEModule`

## Data Format Requirements

- **Input**: numpy.ndarray with dtype `float32` or `float64`
- **Shape**: `(n_samples, n_features)`
- **Output**: matches input type (ndarray in → ndarray out, Tensor in → Tensor out)

## See Also

- [Testing Guide](testing.md)
- [Metrics](metrics.md)
