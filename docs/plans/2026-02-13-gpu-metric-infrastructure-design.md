# GPU Metric Infrastructure Design

**Date:** 2026-02-13
**PR:** #204 (`feat/gpu-metric-optimizations`)
**OpenSpec change:** `add-gpu-metric-infrastructure` (at `openspec/changes/add-gpu-metric-infrastructure/`)
**Status:** Proposal (awaiting approval)

## Overview

Add GPU-accelerated metric computation infrastructure to manyLatents via TorchDR integration. This is a **spec-only** document -- no implementation until approved.

## Audit Findings

### Current State

**LatentModule base** (`algorithms/latent/latent_module_base.py`):
- Constructor: `n_components: int = 2, init_seed: int = 42, **kwargs`
- No `device` or `backend` fields
- `kernel_matrix()` and `affinity_matrix()` return `np.ndarray`, default to `NotImplementedError`
- `affinity_matrix(use_symmetric=True)` calls `symmetric_diffusion_operator(K)` for real eigenvalues

**Algorithm implementations:**

| Module | Library | kernel_matrix() source | affinity_matrix() source |
|--------|---------|----------------------|------------------------|
| UMAPModule | umap-learn | `model.graph_` (sparse) | row-stochastic or `symmetric_diffusion_operator` |
| PHATEModule | phate | `model.graph.K` (sparse) | `model.diff_op` or `symmetric_diffusion_operator` |
| TSNEModule | openTSNE | `affinities.P` (sparse) | row-stochastic or `symmetric_diffusion_operator` |
| DiffusionMapModule | graphtools | `model.G.kernel` (sparse) | `compute_dm()` returns L or S |
| PCAModule | sklearn | `X_centered @ X_centered.T` | Gram / (n-1) |
| MDSModule | custom | double-centered D^2 / (n-1) | same as kernel |

**Existing GPU acceleration:**
- `compute_knn()` in `utils/metrics.py`: FAISS-GPU > FAISS-CPU > sklearn fallback
- `compute_svd_cache()` in `utils/metrics.py`: torch-GPU > numpy-CPU fallback
- Metric dispatch via signature inspection: `_knn_cache`, `_svd_cache` shared

**Metric system:**
- 28+ metrics across 3 levels (dataset, embedding, module)
- `_knn_cache` and `_svd_cache` shared via `inspect.signature()` detection
- AffinitySpectrum calls `module.affinity_matrix(use_symmetric=True)`
- `GroundTruthPreservation` already implements geodesic distance correlation

**Datasets:**
- All synthetic datasets implement `get_gt_dists()`, `get_graph()`, `metadata` (labels)
- Ground truth types: manifold (SwissRoll, Torus), graph (DLAtree), euclidean (GaussianBlobs)
- GaussianBlobs optionally returns `centers`

**Dependencies:**
- No torchdr or faiss in `pyproject.toml`
- torch >= 2.3, scipy >= 1.8 < 1.15
- Optional groups: slurm, dynamics

### TorchDR Source Audit (v0.3, cloned to `/tmp/torchdr-source/`)

**Key API:**
- `DRModule(BaseEstimator, nn.Module, ABC)` base class with `device`, `backend`, `n_components`
- After `fit_transform()`, affinity stored in `self.affinity_in_` (tensor buffer)
- UMAP: `n_neighbors=30`, `backend="faiss"` (default), supports faiss/keops/None
- PHATE: `k=5, t=100, alpha=10.0`, **backend=None only** (no faiss/keops!)
- TSNE: `perplexity=30`, supports faiss/keops/None
- `torchdr.silhouette_score(X, labels, metric, device, backend)`
- `torchdr.neighborhood_preservation(X, Z, K, metric, backend, device)`

**Parameter mapping:**
- UMAP: `n_neighbors` -> `n_neighbors` (same)
- PHATE: `knn` -> `k`, `decay` -> `alpha` (different!)
- TSNE: `perplexity` -> `perplexity` (same)

---

## Design Decisions

### D1: Backend on base class (not mixin)
Add `backend` and `device` to `LatentModule.__init__()`. Simple, non-breaking (defaults to `None`).

### D2: Backend routing via `_create_model()`
Each module implements `_create_model()` returning either CPU library model or TorchDR model.

### D3: Affinity access -- numpy default + `affinity_tensor()` for GPU
- `kernel_matrix()` / `affinity_matrix()` continue returning `np.ndarray`
- New `affinity_tensor()` returns `torch.Tensor` on compute device

### D4: Eigenvalue cache as `_eigenvalue_cache`
Same pattern as `_knn_cache` / `_svd_cache`. Key: `(use_symmetric, top_k)`, value: `np.ndarray`.

### D5: Metrics remain independent (no piping)
Share eigenvalues via cache. No metric-to-metric dependencies.

### D6: Sweeps via Hydra multirun + enhanced merge_results.py

### D7: DatasetCapabilities as runtime discovery function
`get_capabilities(dataset) -> dict[str, bool]` via `hasattr` checks.

---

## Section 1: Unified Backend Architecture

### Changes to `latent_module_base.py`

```python
class LatentModule(ABC):
    def __init__(self, n_components=2, init_seed=42, backend=None, device=None, **kwargs):
        # backend: None | "torchdr" | "sklearn" | "auto"
        # device: None | "cpu" | "cuda" | "auto"
        self.backend = backend
        self.device = device
        ...

    def affinity_tensor(self) -> torch.Tensor:
        """Return affinity as GPU tensor (TorchDR) or converted numpy."""
        if self.backend == "torchdr" and hasattr(self.model, 'affinity_in_'):
            return self.model.affinity_in_
        return torch.from_numpy(self.affinity_matrix(use_symmetric=True))
```

### Changes per algorithm module

Each module gains `_create_model()` with backend routing:

```python
# UMAPModule._create_model()
if self.backend == "torchdr":
    from torchdr import UMAP
    return UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                n_components=self.n_components, device=self.device or "auto")
else:
    from umap import UMAP
    return UMAP(n_neighbors=self.n_neighbors, ...)
```

### New Hydra configs

```yaml
# configs/algorithms/latent/umap_torchdr.yaml
_target_: manylatents.algorithms.latent.umap.UMAPModule
n_components: 2
random_state: ${seed}
n_neighbors: 15
min_dist: 0.5
backend: torchdr
device: auto
```

---

## Section 2: Standalone AffinityModule

```python
# manylatents/algorithms/latent/affinity.py
class AffinityModule(LatentModule):
    def __init__(self, n_components=2, knn=15, alpha=1.0,
                 symmetric=True, metric="euclidean", backend=None, device=None, **kwargs):
        super().__init__(n_components=n_components, backend=backend, device=device, **kwargs)
        self.knn = knn
        self.alpha = alpha
        self.symmetric = symmetric
        self.metric = metric

    def fit(self, x, y=None):
        # Build affinity from input data
        ...
        self._is_fitted = True

    def transform(self, x):
        return x  # identity

    def kernel_matrix(self, ignore_diagonal=False):
        return self._kernel  # N x N numpy

    def affinity_matrix(self, ignore_diagonal=False, use_symmetric=False):
        if use_symmetric:
            return symmetric_diffusion_operator(self._kernel, self.alpha)
        return self._affinity
```

---

## Section 3: Metrics Architecture

### New `_eigenvalue_cache` in `experiment.py:evaluate()`

```python
# After kNN and SVD caches, add:
eigenvalue_cache = None
if module is not None:
    eig_k_values = set()
    for metric_cfg in metric_cfgs.values():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        sig = inspect.signature(metric_fn)
        if "_eigenvalue_cache" in sig.parameters:
            eig_k_values.add(...)
    if eig_k_values:
        eigenvalue_cache = _compute_eigenvalue_cache(module, eig_k_values)
```

### New metrics

| Metric | File | Type | Dependencies |
|--------|------|------|-------------|
| SpectralGapRatio | `metrics/spectral_gap_ratio.py` | module | `_eigenvalue_cache` |
| SpectralDecayRate | `metrics/spectral_decay_rate.py` | module | `_eigenvalue_cache` |
| SilhouetteScore | `metrics/silhouette.py` | embedding | `dataset.metadata` |
| GeodesicDistanceCorrelation | `metrics/geodesic_distance_correlation.py` | dataset | `dataset.get_gt_dists()` |
| MetricAgreement | `metrics/metric_agreement.py` | post-hoc | sweep results DataFrame |
| DatasetTopologyDescriptor | `metrics/dataset_topology_descriptor.py` | dataset+module | affinity + capabilities |

### Metric signatures

```python
def SpectralGapRatio(embeddings, dataset=None, module=None,
                     _eigenvalue_cache=None) -> float:
    ...

def SilhouetteScore(embeddings, dataset=None, module=None,
                    metric="euclidean") -> float:
    ...

def DatasetTopologyDescriptor(embeddings, dataset=None, module=None,
                              _eigenvalue_cache=None) -> dict:
    ...
```

---

## Section 4: Sweep Infrastructure

### New sweep configs

```yaml
# configs/sweep/dataset_algorithm_grid.yaml
defaults:
  - override /metrics/embedding: [trustworthiness, continuity, knn_preservation]
  - override /metrics/module: [affinity_spectrum]

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      data: swissroll,torus,saddle_surface,gaussian_blobs,dla_tree
      algorithms/latent: pca,umap,phate,tsne,diffusionmap
      seed: 42,43,44
```

```yaml
# configs/sweep/umap_parameter_sensitivity.yaml
defaults:
  - override /algorithms/latent: umap
  - override /data: swissroll

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      algorithms.latent.n_neighbors: 5,10,15,30,50
      algorithms.latent.min_dist: 0.01,0.1,0.5,1.0
      seed: 42,43,44
```

---

## Section 5: Dataset Ground Truth Interface

```python
# manylatents/data/capabilities.py
from typing import Any, Protocol

class DatasetCapabilities(Protocol):
    def get_gt_dists(self) -> Any: ...
    def get_graph(self) -> Any: ...
    def get_labels(self) -> Any: ...

def get_capabilities(dataset) -> dict[str, bool | str]:
    caps = {
        "gt_dists": hasattr(dataset, 'get_gt_dists') and callable(dataset.get_gt_dists),
        "graph": hasattr(dataset, 'get_graph') and callable(dataset.get_graph),
        "labels": hasattr(dataset, 'get_labels') and callable(dataset.get_labels),
        "centers": hasattr(dataset, 'get_centers') and callable(dataset.get_centers),
    }
    # Classify ground truth type
    if caps["gt_dists"]:
        cls_name = type(dataset).__name__
        if "DLA" in cls_name or "Tree" in cls_name:
            caps["gt_type"] = "graph"
        elif "Blob" in cls_name:
            caps["gt_type"] = "euclidean"
        else:
            caps["gt_type"] = "manifold"
    else:
        caps["gt_type"] = "unknown"
    return caps
```

---

## Section 6: Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
torchdr = [
    "torchdr>=0.3,<0.4",
    "faiss-cpu",
]
```

```python
# manylatents/utils/backend.py
_torchdr_available = None

def check_torchdr_available() -> bool:
    global _torchdr_available
    if _torchdr_available is None:
        try:
            import torchdr
            _torchdr_available = True
        except ImportError:
            _torchdr_available = False
    return _torchdr_available
```

---

## Section 7: Verification Tests

| Test file | What it verifies |
|-----------|-----------------|
| `tests/test_backend_switching.py` | backend=None vs torchdr produce comparable results (rtol=1e-2) |
| `tests/test_affinity_module.py` | AffinityModule fit/transform/kernel_matrix/affinity_matrix |
| `tests/test_eigenvalue_cache.py` | Cache computation and sharing across metrics |
| `tests/test_new_metrics.py` | Each new metric on SwissRoll synthetic data |
| `tests/test_dataset_capabilities.py` | `get_capabilities()` on all synthetic datasets |
| `tests/test_sweep_configs.py` | Dry-run sweep configs with `--cfg job` |

All TorchDR-dependent tests gated with:
```python
@pytest.mark.skipif(not check_torchdr_available(), reason="torchdr not installed")
```
