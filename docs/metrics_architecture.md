# Metrics Architecture

## Overview

manylatents uses a **three-level metrics architecture** to evaluate different aspects of the dimensionality reduction pipeline. This separation ensures metrics are computed at the appropriate stage and provides fine-grained control over evaluation.

## The Three Metric Levels

### 1. Dataset Metrics (`metrics/dataset/`)

**Purpose**: Evaluate properties of the **original high-dimensional data**.

**When Executed**:
- At the beginning of a run (before dimensionality reduction)
- Optionally at the end (to verify data hasn't been corrupted)

**Use Cases**:
- Data quality checks (missing values, outliers)
- Intrinsic dimensionality estimation
- Population structure analysis (e.g., admixture, stratification)
- Ground truth preservation metrics

**Examples**:
- `stratification.py`: Measures population structure in genetic data
- `gt_preservation.py`: Checks if known ground truth relationships are preserved
- `sample_id.py`: Verifies sample identity across transformations

**Why Separate?**: Dataset metrics are independent of the DR algorithm and only need to be computed once per dataset, not once per algorithm.

---

### 2. Embedding Metrics (`metrics/embedding/`)

**Purpose**: Evaluate the **quality of the low-dimensional embeddings** produced by DR algorithms.

**When Executed**:
- After each dimensionality reduction step
- When embeddings are generated or loaded

**Use Cases**:
- Neighborhood preservation (trustworthiness, continuity)
- Geometric properties (intrinsic dimensionality, fractal dimension)
- Topology preservation (persistent homology)
- Distance preservation (kNN preservation, tangent space alignment)

**Examples**:
- `trustworthiness.py`: Measures how well local neighborhoods are preserved
- `persistent_homology.py`: Analyzes topological features across scales
- `lid.py`: Estimates local intrinsic dimensionality of embeddings
- `knn_preservation.py`: Checks if k-nearest neighbors are maintained

**Why Separate?**: Embedding metrics compare high-dimensional vs. low-dimensional representations. They are the core evaluation for DR algorithms and need access to both original data and embeddings.

---

### 3. Module Metrics (`metrics/module/`)

**Purpose**: Evaluate **algorithm-specific internal components** (e.g., learned graphs, kernels, decompositions).

**When Executed**:
- During model training (potentially every iteration)
- After model fitting (for post-hoc analysis)

**Use Cases**:
- Graph connectivity (number of connected components)
- Kernel/affinity matrix properties (spectrum, sparsity)
- Learned parameters (eigenvalues, loadings)
- Convergence diagnostics

**Examples**:
- `affinity_spectrum.py`: Analyzes eigenvalues of affinity matrices
- `connected_components.py`: Counts connected components in learned graphs

**Why Separate?**: Module metrics are algorithm-specific and may not be applicable to all DR methods. They require access to internal model state (the `module` parameter) rather than just embeddings.

---

## Metric Return Types

All metrics must return one of three types (defined in `metrics/metric.py`):

```python
Union[float, tuple[float, np.ndarray], dict[str, Any]]
```

### 1. `float` - Simple Scalar
```python
def Trustworthiness(...) -> float:
    return 0.95
```
Use for: Basic summary statistics

### 2. `tuple[float, np.ndarray]` - Scalar + Per-Sample
```python
def TestMetric(...) -> tuple[float, np.ndarray]:
    per_sample = np.zeros(n_samples)
    return (0.0, per_sample)
```
Use for: Metrics with per-sample breakdowns (enables wandb table logging)

### 3. `dict[str, Any]` - Structured Output
```python
def ReebGraphNodesEdges(...) -> dict[str, Any]:
    return {'nodes': nodes, 'edges': edges}
```
Use for: Complex metrics (graphs, histograms, multiple values)

---

## Configuration Structure

Metrics are organized hierarchically in Hydra configs:

```yaml
# configs/metrics/example.yaml
defaults:
  - dataset:
    - stratification      # Run at start
  - embedding:
    - trustworthiness     # Run after DR
    - persistent_homology
  - module:
    - affinity_spectrum   # Run on model internals
  - _self_
```

This allows:
- **Selective evaluation**: Only run metrics for specific levels
- **Compositional configs**: Mix and match metrics across levels
- **Clear separation**: Easy to understand when each metric runs

---

## Design Philosophy

### Why Three Levels?

1. **Efficiency**: Dataset metrics run once, not per algorithm
2. **Clarity**: Clear separation of concerns (data vs. embeddings vs. model)
3. **Flexibility**: Different DR methods may not have comparable modules
4. **Optimization**: Can selectively disable expensive metrics per level

### Example Workflow

```python
# Beginning of pipeline
dataset_metrics = compute_metrics(data, cfg.metrics.dataset)
# → Runs once, caches results

# After PCA
pca_embeddings = pca.transform(data)
embedding_metrics = compute_metrics(pca_embeddings, cfg.metrics.embedding)
# → Compares data vs. pca_embeddings

# After UMAP (in multi-step pipeline)
umap_embeddings = umap.transform(pca_embeddings)
embedding_metrics = compute_metrics(umap_embeddings, cfg.metrics.embedding)
# → Compares pca_embeddings vs. umap_embeddings

# Model-specific analysis
module_metrics = compute_metrics(umap.graph_, cfg.metrics.module)
# → Analyzes UMAP's learned kNN graph
```

---

## Best Practices for Metric Authors

### Choosing the Right Level

- **Dataset**: Does it only need original data? → `metrics/dataset/`
- **Embedding**: Does it compare original vs. reduced? → `metrics/embedding/`
- **Module**: Does it need algorithm internals? → `metrics/module/`

### Function Signature

All metrics must accept:
```python
def YourMetric(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    **kwargs  # Additional hyperparameters
) -> Union[float, tuple[float, np.ndarray], dict[str, Any]]:
    ...
```

- `embeddings`: The output data (for embedding metrics) or original data (for dataset metrics)
- `dataset`: Access to original data (has `.data` attribute)
- `module`: Access to algorithm internals (e.g., `.graph_`, `.kernel_`)

### Configuration

Create a YAML config in the appropriate subdirectory:

```yaml
# configs/metrics/embedding/your_metric.yaml
your_metric:
  _target_: manylatents.metrics.your_metric.YourMetric
  _partial_: True
  k: 25
  threshold: 0.5
```

---

## Testing Your Metrics

Use `test_metric` to verify your metric integrates correctly:

```yaml
# configs/metrics/test_metric.yaml
defaults:
  - dataset:
    - test_metric
  - embedding:
    - test_metric
  - module:
    - test_metric
```

This runs a fast (instant) metric across all three levels to verify:
- Configuration is correct
- All levels are computed
- Wandb logging works (scalar + tables)

---

## Future Extensions

The three-level architecture is designed to be extensible:

- **Pipeline-level metrics**: Evaluate entire multi-step workflows
- **Comparative metrics**: Compare multiple algorithms side-by-side
- **Temporal metrics**: Track metric evolution during training
- **Meta-metrics**: Metrics that aggregate other metrics

All can be added without breaking the core three-level structure.
