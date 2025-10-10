# API Usage Guide

The manyLatents programmatic API enables agent-driven workflows and in-memory data chaining, allowing you to compose dimensionality reduction pipelines programmatically.

## Quick Start

```python
from manylatents.api import run

# Single algorithm run
result = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 10}}
)

embeddings = result['embeddings']
```

## Chained Workflows

The API supports chaining multiple algorithms by passing the output of one as input to another:

```python
import numpy as np
import torch

def ensure_numpy(arr):
    """Helper to convert tensor to numpy."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

# Step 1: Initial dimensionality reduction
result1 = run(
    data='swissroll',
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.pca.PCAModule', 'n_components': 50}}
)

# Step 2: Chain to another algorithm
result2 = run(
    input_data=ensure_numpy(result1['embeddings']),
    algorithms={'latent': {'_target_': 'manylatents.algorithms.latent.umap.UMAPModule', 'n_components': 2}}
)

final_embeddings = ensure_numpy(result2['embeddings'])
```

## Available Algorithms

### Dimensionality Reduction

- **PCA**: `manylatents.algorithms.latent.pca.PCAModule`
- **t-SNE**: `manylatents.algorithms.latent.tsne.TSNEModule`
- **UMAP**: `manylatents.algorithms.latent.umap.UMAPModule`
- **PHATE**: `manylatents.algorithms.latent.phate.PHATEModule`

## API Reference

### `run(input_data=None, **overrides)`

Execute a dimensionality reduction algorithm.

**Parameters:**

- `input_data` (np.ndarray, optional): In-memory data array of shape `(n_samples, n_features)`. If provided, this data is used instead of loading from a dataset.
- `**overrides`: Configuration overrides (e.g., `data='swissroll'`, `algorithms={...}`)

**Returns:**

Dictionary with keys:
- `embeddings`: Computed embeddings (torch.Tensor or np.ndarray)
- `label`: Labels from dataset (if available)
- `metadata`: Run metadata dictionary
- `scores`: Evaluation metrics (if enabled)

**Examples:**

```python
# Using a built-in dataset
result = run(data='swissroll', algorithms={'latent': 'pca'})

# Using in-memory data
my_data = np.random.randn(1000, 100).astype(np.float32)
result = run(input_data=my_data, algorithms={'latent': 'pca'})

# Chaining algorithms
result2 = run(input_data=ensure_numpy(result['embeddings']), algorithms={'latent': 'umap'})
```

## Advanced Usage

### Custom Configuration

```python
result = run(
    data='swissroll',
    algorithms={
        'latent': {
            '_target_': 'manylatents.algorithms.latent.umap.UMAPModule',
            'n_components': 2,
            'n_neighbors': 15,
            'min_dist': 0.1
        }
    },
    data={
        'batch_size': 256,
        'num_workers': 4
    }
)
```

### Disabling Features for Speed

```python
# Disable W&B logging
result = run(data='swissroll', algorithms={'latent': 'pca'}, debug=True)

# Skip evaluation metrics
result = run(data='swissroll', algorithms={'latent': 'pca'}, metrics=None)
```

## Data Format Requirements

- **Input**: numpy.ndarray with dtype `float32` or `float64`
- **Shape**: `(n_samples, n_features)`
- **Output**: May be torch.Tensor or numpy.ndarray (use `ensure_numpy()` helper for consistency)

## Multi-Step Example

```python
from manylatents.api import run
import numpy as np
import torch

def ensure_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

# Progressive dimensionality reduction pipeline
steps = [
    ('PCA 100D', 'manylatents.algorithms.latent.pca.PCAModule', 100),
    ('PCA 50D', 'manylatents.algorithms.latent.pca.PCAModule', 50),
    ('UMAP 2D', 'manylatents.algorithms.latent.umap.UMAPModule', 2),
]

# Initial data
current_data = run(data='swissroll', algorithms={'latent': {'_target_': steps[0][1], 'n_components': steps[0][2]}})

# Chain subsequent steps
for name, target, n_comp in steps[1:]:
    print(f"Running {name}...")
    current_data = run(
        input_data=ensure_numpy(current_data['embeddings']),
        algorithms={'latent': {'_target_': target, 'n_components': n_comp}}
    )

final_embeddings = ensure_numpy(current_data['embeddings'])
print(f"Final shape: {final_embeddings.shape}")
```

## Implementation Details

The in-memory data pipeline uses:
- `PrecomputedDataModule`: Accepts `data` parameter for numpy arrays
- `InMemoryDataset`: Wraps arrays in EmbeddingOutputs format
- Compatible with all manyLatents metrics, callbacks, and visualizations

## Troubleshooting

### Common Issues

**"PrecomputedDataModule requires either a 'path' or 'data' argument"**

Provide either `data='dataset_name'` OR `input_data=array`, not both or neither.

**Tensor/NumPy Conversion**

Use the helper function:

```python
def ensure_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr
```

## See Also

- [Testing Guide](testing.md)
- Configuration system documentation (coming soon)
- Algorithm-specific parameters (coming soon)
