# API Usage Guide

The ManyLatents programmatic API enables workflow integration and in-memory data chaining.

## Quick Start

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

## Chained Workflows

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

Note: Embeddings are automatically converted to numpy arrays by the evaluation system.

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
- `embeddings`: Computed embeddings (numpy array)
- `label`: Labels from dataset (if available)
- `metadata`: Run metadata dictionary
- `scores`: Evaluation metrics (if enabled)

**Examples:**

```python
# Using a built-in dataset
result = run(data='swissroll', algorithms={'latent': 'pca'})

# Using in-memory data
import numpy as np
my_data = np.random.randn(1000, 100).astype(np.float32)
result = run(input_data=my_data, algorithms={'latent': 'pca'})

# Chaining algorithms
result2 = run(input_data=result['embeddings'], algorithms={'latent': 'umap'})
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
- **Output**: numpy array (tensor conversion is handled automatically)

## Multi-Step Example

```python
from manylatents.api import run

# Progressive dimensionality reduction
steps = [
    ('PCA 100D', 'manylatents.algorithms.latent.pca.PCAModule', 100),
    ('PCA 50D', 'manylatents.algorithms.latent.pca.PCAModule', 50),
    ('UMAP 2D', 'manylatents.algorithms.latent.umap.UMAPModule', 2),
]

# Initial data
current_data = run(
    data='swissroll',
    algorithms={'latent': {'_target_': steps[0][1], 'n_components': steps[0][2]}}
)

# Chain subsequent steps
for name, target, n_comp in steps[1:]:
    print(f"Running {name}...")
    current_data = run(
        input_data=current_data['embeddings'],
        algorithms={'latent': {'_target_': target, 'n_components': n_comp}}
    )

print(f"Final shape: {current_data['embeddings'].shape}")
```

## Implementation Details

The in-memory data pipeline uses:
- `PrecomputedDataModule`: Accepts `data` parameter for numpy arrays
- `InMemoryDataset`: Wraps arrays in EmbeddingOutputs format
- Compatible with all ManyLatents metrics, callbacks, and visualizations

## Troubleshooting

### Common Issues

**"PrecomputedDataModule requires either a 'path' or 'data' argument"**

Provide either `data='dataset_name'` OR `input_data=array`, not both or neither.

## See Also

- [Testing Guide](testing.md)
- [Metrics](metrics.md)
