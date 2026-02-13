# GPU Metric Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-accelerated metric computation infrastructure to manyLatents via TorchDR integration, new spectral metrics, dataset capabilities discovery, and sweep infrastructure.

**Architecture:** Backend routing on `LatentModule` base class (`backend`/`device` params) dispatches to either existing CPU libraries or TorchDR GPU equivalents. New `_eigenvalue_cache` shares spectral decomposition across metrics. `get_capabilities()` provides runtime dataset discovery. Sweep configs enable systematic benchmarking.

**Tech Stack:** TorchDR v0.3, FAISS-CPU, PyTorch eigvalsh, Hydra multirun, scipy/numpy

**Design doc:** `docs/plans/2026-02-13-gpu-metric-infrastructure-design.md`
**OpenSpec:** `openspec/changes/add-gpu-metric-infrastructure/`

---

## Task 1: Dependencies and Import Guards

**Files:**
- Modify: `pyproject.toml:65-69`
- Create: `manylatents/utils/backend.py`
- Test: `tests/test_backend_utils.py`

**Step 1: Write failing tests for import guard utilities**

Create `tests/test_backend_utils.py`:

```python
"""Tests for backend availability utilities."""
import pytest


def test_check_torchdr_available_returns_bool():
    """check_torchdr_available returns a boolean."""
    from manylatents.utils.backend import check_torchdr_available

    result = check_torchdr_available()
    assert isinstance(result, bool)


def test_check_torchdr_available_caches_result():
    """Subsequent calls use cached result without re-importing."""
    from manylatents.utils import backend

    # Reset cache
    backend._torchdr_available = None
    first = backend.check_torchdr_available()
    # Set to opposite to prove cache is used
    backend._torchdr_available = not first
    second = backend.check_torchdr_available()
    assert second == (not first)
    # Clean up
    backend._torchdr_available = None


def test_check_faiss_available_returns_bool():
    """check_faiss_available returns a boolean."""
    from manylatents.utils.backend import check_faiss_available

    result = check_faiss_available()
    assert isinstance(result, bool)


def test_resolve_device_returns_string():
    """resolve_device returns a valid device string."""
    from manylatents.utils.backend import resolve_device

    result = resolve_device(None)
    assert result in ("cpu", "cuda")

    result_cpu = resolve_device("cpu")
    assert result_cpu == "cpu"


def test_resolve_backend_none():
    """resolve_backend with None returns None."""
    from manylatents.utils.backend import resolve_backend

    assert resolve_backend(None) is None


def test_resolve_backend_sklearn():
    """resolve_backend with 'sklearn' returns None (CPU library)."""
    from manylatents.utils.backend import resolve_backend

    assert resolve_backend("sklearn") is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_backend_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'manylatents.utils.backend'`

**Step 3: Implement `manylatents/utils/backend.py`**

```python
"""Backend availability checks and resolution utilities.

Provides cached import guards for optional dependencies (TorchDR, FAISS)
and device/backend resolution helpers.
"""
import logging

import torch

logger = logging.getLogger(__name__)

_torchdr_available = None
_faiss_available = None


def check_torchdr_available() -> bool:
    """Check if TorchDR is importable. Result is cached after first call."""
    global _torchdr_available
    if _torchdr_available is None:
        try:
            import torchdr  # noqa: F401

            _torchdr_available = True
        except ImportError:
            _torchdr_available = False
    return _torchdr_available


def check_faiss_available() -> bool:
    """Check if FAISS is importable. Result is cached after first call."""
    global _faiss_available
    if _faiss_available is None:
        try:
            import faiss  # noqa: F401

            _faiss_available = True
        except ImportError:
            _faiss_available = False
    return _faiss_available


def resolve_device(device: str | None) -> str:
    """Resolve device string to concrete device.

    Args:
        device: None, "cpu", "cuda", or "auto".

    Returns:
        "cpu" or "cuda".
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_backend(backend: str | None) -> str | None:
    """Resolve backend string.

    Args:
        backend: None, "sklearn", "torchdr", or "auto".

    Returns:
        None (use CPU library) or "torchdr".

    Raises:
        ImportError: If "torchdr" requested but not installed.
    """
    if backend is None or backend == "sklearn":
        return None
    if backend == "torchdr":
        if not check_torchdr_available():
            raise ImportError(
                "TorchDR backend requested but not installed. "
                "Install with: pip install manylatents[torchdr]"
            )
        return "torchdr"
    if backend == "auto":
        if check_torchdr_available() and torch.cuda.is_available():
            return "torchdr"
        return None
    raise ValueError(f"Unknown backend: {backend!r}. Use None, 'sklearn', 'torchdr', or 'auto'.")
```

**Step 4: Run tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_backend_utils.py -v`
Expected: PASS (all 6 tests)

**Step 5: Add torchdr optional dependency group to `pyproject.toml`**

Add after the `dynamics` line (line 69):

```toml
# GPU-accelerated DR via TorchDR
torchdr = [
    "torchdr>=0.3,<0.4",
    "faiss-cpu",
]
```

**Step 6: Commit**

```bash
git add manylatents/utils/backend.py tests/test_backend_utils.py pyproject.toml
git commit -m "feat: add backend utility module and torchdr optional dependency group"
```

---

## Task 2: Dataset Capabilities Discovery

**Files:**
- Create: `manylatents/data/capabilities.py`
- Test: `tests/test_dataset_capabilities.py`

**Step 1: Write failing tests**

Create `tests/test_dataset_capabilities.py`:

```python
"""Tests for dataset capability discovery."""
import numpy as np
import pytest


class FakeDatasetWithAll:
    """Fake dataset with all ground truth methods."""
    def get_gt_dists(self):
        return np.eye(10)

    def get_graph(self):
        return "graph"

    def get_labels(self):
        return np.arange(10)

    def get_centers(self):
        return np.zeros((3, 2))


class FakeDatasetMinimal:
    """Fake dataset with no ground truth."""
    pass


class FakeSwissRoll:
    """Fake SwissRoll-like dataset."""
    def get_gt_dists(self):
        return np.eye(10)

    def get_graph(self):
        return "graph"

    def get_labels(self):
        return np.arange(10)


class FakeDLATree:
    """Fake DLAtree-like dataset."""
    def get_gt_dists(self):
        return np.eye(10)

    def get_graph(self):
        return "graph"

    def get_labels(self):
        return np.arange(10)


class FakeGaussianBlobs:
    """Fake GaussianBlobs-like dataset."""
    def get_gt_dists(self):
        return np.eye(10)

    def get_graph(self):
        return "graph"

    def get_labels(self):
        return np.arange(10)

    def get_centers(self):
        return np.zeros((3, 2))


def test_get_capabilities_full_dataset():
    from manylatents.data.capabilities import get_capabilities

    caps = get_capabilities(FakeDatasetWithAll())
    assert caps["gt_dists"] is True
    assert caps["graph"] is True
    assert caps["labels"] is True
    assert caps["centers"] is True


def test_get_capabilities_minimal_dataset():
    from manylatents.data.capabilities import get_capabilities

    caps = get_capabilities(FakeDatasetMinimal())
    assert caps["gt_dists"] is False
    assert caps["graph"] is False
    assert caps["labels"] is False
    assert caps["centers"] is False
    assert caps["gt_type"] == "unknown"


def test_gt_type_manifold():
    from manylatents.data.capabilities import get_capabilities

    caps = get_capabilities(FakeSwissRoll())
    # Default gt_type for non-DLA, non-Blob datasets with gt_dists
    assert caps["gt_type"] == "manifold"


def test_gt_type_graph():
    from manylatents.data.capabilities import get_capabilities

    ds = FakeDLATree()
    ds.__class__.__name__ = "DLATreeFromGraph"
    # Rename to trigger DLA detection
    type(ds).__qualname__ = "DLATreeFromGraph"
    # Use a simpler approach: pass an object whose class name contains "DLA"
    class DLATreeDataset:
        def get_gt_dists(self): return np.eye(5)
        def get_graph(self): return "g"
        def get_labels(self): return np.arange(5)

    caps = get_capabilities(DLATreeDataset())
    assert caps["gt_type"] == "graph"


def test_gt_type_euclidean():
    from manylatents.data.capabilities import get_capabilities

    class BlobsDataset:
        def get_gt_dists(self): return np.eye(5)
        def get_graph(self): return "g"
        def get_labels(self): return np.arange(5)
        def get_centers(self): return np.zeros((3, 2))

    caps = get_capabilities(BlobsDataset())
    assert caps["gt_type"] == "euclidean"
```

**Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_dataset_capabilities.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `manylatents/data/capabilities.py`**

```python
"""Dataset capability discovery.

Provides runtime inspection of dataset ground truth interfaces.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DatasetCapabilities(Protocol):
    """Protocol for datasets with optional ground truth methods."""

    def get_gt_dists(self) -> Any: ...
    def get_graph(self) -> Any: ...
    def get_labels(self) -> Any: ...


def get_capabilities(dataset: Any) -> dict[str, bool | str]:
    """Inspect a dataset and return which ground truth interfaces it supports.

    Args:
        dataset: Any dataset object.

    Returns:
        Dict with keys: gt_dists, graph, labels, centers (bool),
        and gt_type (str: "manifold"|"graph"|"euclidean"|"unknown").
    """
    caps: dict[str, bool | str] = {
        "gt_dists": hasattr(dataset, "get_gt_dists") and callable(dataset.get_gt_dists),
        "graph": hasattr(dataset, "get_graph") and callable(dataset.get_graph),
        "labels": hasattr(dataset, "get_labels") and callable(dataset.get_labels),
        "centers": hasattr(dataset, "get_centers") and callable(dataset.get_centers),
    }

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


def log_capabilities(dataset: Any) -> dict[str, bool | str]:
    """Discover and log dataset capabilities."""
    caps = get_capabilities(dataset)
    logger.info(
        "Dataset capabilities: "
        + ", ".join(f"{k}={v}" for k, v in caps.items())
    )
    return caps
```

**Step 4: Run tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_dataset_capabilities.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add manylatents/data/capabilities.py tests/test_dataset_capabilities.py
git commit -m "feat: add dataset capabilities discovery module"
```

---

## Task 3: LatentModule Base Class — Backend and Device Parameters

**Files:**
- Modify: `manylatents/algorithms/latent/latent_module_base.py`
- Test: `tests/test_latent_module_backend.py`

**Step 1: Write failing tests**

Create `tests/test_latent_module_backend.py`:

```python
"""Tests for LatentModule backend/device parameters."""
import numpy as np
import pytest
import torch


class ConcreteModule:
    """Minimal concrete LatentModule for testing base class features."""
    pass


def test_latent_module_accepts_backend_param():
    """LatentModule.__init__ accepts backend parameter."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x

    m = TestModule(n_components=2, backend="torchdr", device="cpu")
    assert m.backend == "torchdr"
    assert m.device == "cpu"


def test_latent_module_defaults_none():
    """Backend and device default to None."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x

    m = TestModule()
    assert m.backend is None
    assert m.device is None


def test_affinity_tensor_from_numpy():
    """affinity_tensor() converts numpy affinity to torch.Tensor."""
    from manylatents.algorithms.latent.latent_module_base import LatentModule

    class TestModule(LatentModule):
        def fit(self, x, y=None): pass
        def transform(self, x): return x
        def affinity_matrix(self, ignore_diagonal=False, use_symmetric=False):
            return np.eye(5, dtype=np.float64)

    m = TestModule()
    t = m.affinity_tensor()
    assert isinstance(t, torch.Tensor)
    assert t.shape == (5, 5)


def test_existing_modules_unaffected():
    """Existing modules still work without backend/device."""
    from manylatents.algorithms.latent.umap import UMAPModule

    # Should not raise — backend/device go to **kwargs -> base class
    m = UMAPModule(n_components=2, random_state=42)
    assert m.backend is None
    assert m.device is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_latent_module_backend.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'backend'`

**Step 3: Modify `latent_module_base.py`**

In `manylatents/algorithms/latent/latent_module_base.py`, change the `__init__` and add `affinity_tensor()`:

```python
# Replace __init__ signature (line 8):
    def __init__(self, n_components: int = 2, init_seed: int = 42,
                 backend: str | None = None, device: str | None = None, **kwargs):
        """Base class for latent modules (DR, clustering, etc.)."""
        self.n_components = n_components
        self.init_seed = init_seed
        self.backend = backend
        self.device = device
        # Flexible handling: if datamodule is passed, store it as a weak port
        self.datamodule = kwargs.pop('datamodule', None)
        # Ignore any other unexpected kwargs to maintain compatibility
        self._is_fitted = False
```

Add `affinity_tensor()` method after `affinity_matrix()` (after line 84):

```python
    def affinity_tensor(self) -> 'torch.Tensor':
        """Return affinity matrix as a torch.Tensor.

        When the TorchDR backend is active, returns the GPU tensor directly.
        Otherwise, converts the numpy affinity matrix.

        Returns:
            torch.Tensor: Affinity matrix on compute device.
        """
        import torch

        if self.backend == "torchdr" and hasattr(self, 'model') and hasattr(self.model, 'affinity_in_'):
            return self.model.affinity_in_
        return torch.from_numpy(
            self.affinity_matrix(use_symmetric=True).astype('float32')
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_latent_module_backend.py -v`
Expected: PASS

**Step 5: Verify existing tests still pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/ -v --ignore=tests/.venv -x`
Expected: PASS (no regressions)

**Step 6: Commit**

```bash
git add manylatents/algorithms/latent/latent_module_base.py tests/test_latent_module_backend.py
git commit -m "feat: add backend/device params and affinity_tensor() to LatentModule base"
```

---

## Task 4: UMAPModule TorchDR Backend

**Files:**
- Modify: `manylatents/algorithms/latent/umap.py`
- Create: `manylatents/configs/algorithms/latent/umap_torchdr.yaml`
- Modify: `manylatents/configs/algorithms/latent/umap.yaml`
- Test: `tests/test_umap_backend.py`

**Step 1: Write failing tests**

Create `tests/test_umap_backend.py`:

```python
"""Tests for UMAPModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_umap_default_backend_unchanged():
    """UMAPModule with default backend uses umap-learn."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, n_neighbors=5, n_epochs=10)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    emb = m.transform(x)
    assert emb.shape == (50, 2)
    # Should have umap-learn model
    from umap import UMAP as UmapLearnUMAP
    assert isinstance(m.model, UmapLearnUMAP)


def test_umap_accepts_backend_param():
    """UMAPModule accepts backend/device without error."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_umap_torchdr_backend_fit_transform():
    """UMAPModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(
        n_components=2, random_state=42, n_neighbors=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_umap_torchdr_affinity_tensor():
    """UMAPModule with torchdr backend exposes affinity_tensor()."""
    from manylatents.algorithms.latent.umap import UMAPModule

    m = UMAPModule(
        n_components=2, random_state=42, n_neighbors=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    m.fit_transform(x)
    t = m.affinity_tensor()
    assert isinstance(t, torch.Tensor)


def test_umap_torchdr_not_installed_raises():
    """UMAPModule with torchdr backend raises if not installed."""
    import manylatents.utils.backend as backend_mod
    from manylatents.algorithms.latent.umap import UMAPModule

    # Temporarily fake torchdr as unavailable
    original = backend_mod._torchdr_available
    backend_mod._torchdr_available = False

    try:
        with pytest.raises(ImportError, match="torchdr"):
            UMAPModule(
                n_components=2, random_state=42,
                backend="torchdr", device="cpu",
            )
    finally:
        backend_mod._torchdr_available = original
```

**Step 2: Run tests to verify they fail**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_umap_backend.py -v`
Expected: Some tests FAIL (torchdr backend not implemented yet)

**Step 3: Modify `manylatents/algorithms/latent/umap.py`**

Replace the entire file with backend-aware version:

```python
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator
from ...utils.backend import resolve_backend, resolve_device


class UMAPModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.5,
        metric: str = 'euclidean',
        n_epochs: Optional[int] = 200,
        learning_rate: float = 1.0,
        fit_fraction: float = 1.0,
        backend: str | None = None,
        device: str | None = None,
        **kwargs
    ):
        super().__init__(
            n_components=n_components, init_seed=random_state,
            backend=backend, device=device, **kwargs,
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fit_fraction = fit_fraction
        self.random_state = random_state

        self._resolved_backend = resolve_backend(backend)
        self.model = self._create_model()

    def _create_model(self):
        if self._resolved_backend == "torchdr":
            from torchdr import UMAP

            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                device=resolve_device(self.device),
                random_state=self.random_state,
            )
        else:
            from umap import UMAP

            return UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
            )

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits UMAP on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            self.model.fit(x_torch)
        else:
            self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted UMAP model."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            x_torch = th.from_numpy(x_np).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.transform(x_torch)
            return torch.tensor(embedding.cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            embedding = self.model.transform(x_np)
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()

        if self._resolved_backend == "torchdr":
            import torch as th
            n_fit = max(1, int(self.fit_fraction * x_np.shape[0]))
            x_torch = th.from_numpy(x_np[:n_fit]).float()
            if resolve_device(self.device) == "cuda":
                x_torch = x_torch.cuda()
            embedding = self.model.fit_transform(x_torch)
            self._is_fitted = True
            return torch.tensor(embedding.cpu().numpy(), device=x.device, dtype=x.dtype)
        else:
            embedding = self.model.fit_transform(x_np)
            self._is_fitted = True
            return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """Returns UMAP affinity matrix."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if use_symmetric:
            K = self.kernel_matrix(ignore_diagonal=ignore_diagonal)
            return symmetric_diffusion_operator(K)
        else:
            if self._resolved_backend == "torchdr":
                A = self.model.affinity_in_.cpu().numpy()
                if hasattr(A, 'toarray'):
                    A = A.toarray()
                A = np.asarray(A)
            else:
                A = np.asarray(self.model.graph_.todense())

            if ignore_diagonal:
                A = A - np.diag(np.diag(A))
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            return A / row_sums

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """Returns UMAP kernel matrix."""
        if not self._is_fitted:
            raise RuntimeError("UMAP model is not fitted yet. Call `fit` first.")

        if self._resolved_backend == "torchdr":
            K = self.model.affinity_in_.cpu().numpy()
            if hasattr(K, 'toarray'):
                K = K.toarray()
            K = np.asarray(K)
        else:
            K = np.asarray(self.model.graph_.todense())

        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K
```

**Step 4: Update Hydra configs**

Add `backend: null` and `device: null` to `manylatents/configs/algorithms/latent/umap.yaml`:

```yaml
_target_: manylatents.algorithms.latent.umap.UMAPModule
n_components: 2
random_state: ${seed}
n_neighbors: 15
min_dist: 0.5
n_epochs: 500
metric: 'euclidean'
learning_rate: 1.0
fit_fraction: 1.0
backend: null
device: null
```

Create `manylatents/configs/algorithms/latent/umap_torchdr.yaml`:

```yaml
_target_: manylatents.algorithms.latent.umap.UMAPModule
n_components: 2
random_state: ${seed}
n_neighbors: 15
min_dist: 0.5
fit_fraction: 1.0
backend: torchdr
device: auto
```

**Step 5: Run tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_umap_backend.py -v`
Expected: PASS (torchdr tests skipped if not installed)

**Step 6: Commit**

```bash
git add manylatents/algorithms/latent/umap.py manylatents/configs/algorithms/latent/umap.yaml manylatents/configs/algorithms/latent/umap_torchdr.yaml tests/test_umap_backend.py
git commit -m "feat: add TorchDR backend support to UMAPModule"
```

---

## Task 5: PHATEModule TorchDR Backend

**Files:**
- Modify: `manylatents/algorithms/latent/phate.py`
- Create: `manylatents/configs/algorithms/latent/phate_torchdr.yaml`
- Modify: `manylatents/configs/algorithms/latent/phate.yaml`
- Test: `tests/test_phate_backend.py`

**Step 1: Write failing tests**

Create `tests/test_phate_backend.py`:

```python
"""Tests for PHATEModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_phate_default_backend_unchanged():
    """PHATEModule with default backend uses phate library."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, knn=5, t=5)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    from phate import PHATE
    assert isinstance(m.model, PHATE)


def test_phate_accepts_backend_param():
    """PHATEModule accepts backend/device without error."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_backend_fit_transform():
    """PHATEModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=5, t=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_param_mapping():
    """PHATEModule maps knn->k and decay->alpha for TorchDR."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=10, t=50, decay=40,
        backend="torchdr", device="cpu",
    )
    # TorchDR PHATE uses 'k' not 'knn', 'alpha' not 'decay'
    assert m.model.k == 10
    assert m.model.alpha == 40
    assert m.model.t == 50
```

**Step 2: Run to verify failure**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_phate_backend.py -v`

**Step 3: Modify `manylatents/algorithms/latent/phate.py`**

Apply the same backend routing pattern as UMAP. Key differences:
- PHATE TorchDR uses `k` (not `knn`), `alpha` (not `decay`)
- PHATE TorchDR **forces `backend=None`** internally (no faiss/keops support)

Add to `__init__` parameters: `backend: str | None = None, device: str | None = None`

Add `_create_model()` method that creates either `phate.PHATE(...)` or `torchdr.PHATE(k=self.knn, alpha=self.decay, t=self.t, ...)`.

Update `fit()`, `transform()`, `fit_transform()` to route through backend.

**Step 4: Add configs**

Add `backend: null` and `device: null` to `phate.yaml`.

Create `phate_torchdr.yaml`:

```yaml
_target_: manylatents.algorithms.latent.phate.PHATEModule
n_components: 2
random_state: ${seed}
knn: 5
t: 5
gamma: 1.0
decay: 40
fit_fraction: 1.0
backend: torchdr
device: auto
```

**Step 5: Run tests, commit**

```bash
git add manylatents/algorithms/latent/phate.py manylatents/configs/algorithms/latent/phate.yaml manylatents/configs/algorithms/latent/phate_torchdr.yaml tests/test_phate_backend.py
git commit -m "feat: add TorchDR backend support to PHATEModule"
```

---

## Task 6: TSNEModule TorchDR Backend

**Files:**
- Modify: `manylatents/algorithms/latent/tsne.py`
- Create: `manylatents/configs/algorithms/latent/tsne_torchdr.yaml`
- Modify: `manylatents/configs/algorithms/latent/tsne.yaml`
- Test: `tests/test_tsne_backend.py`

Same pattern as Task 4 and 5. TSNE parameter mapping: `perplexity` -> `perplexity` (same name).

**Step 1: Write failing tests** — Same pattern as `test_umap_backend.py`

**Step 2: Run to verify failure**

**Step 3: Modify `tsne.py`** — Add `backend`/`device` params, `_create_model()`, route `fit()`/`transform()`/`fit_transform()` through backend.

**Step 4: Add configs** — `tsne.yaml` gets `backend: null, device: null`. Create `tsne_torchdr.yaml`.

**Step 5: Run tests, commit**

```bash
git add manylatents/algorithms/latent/tsne.py manylatents/configs/algorithms/latent/tsne.yaml manylatents/configs/algorithms/latent/tsne_torchdr.yaml tests/test_tsne_backend.py
git commit -m "feat: add TorchDR backend support to TSNEModule"
```

---

## Task 7: Eigenvalue Cache in experiment.py

**Files:**
- Modify: `manylatents/experiment.py:230-275` (inside `evaluate_embeddings`)
- Test: `tests/test_eigenvalue_cache.py`

**Step 1: Write failing tests**

Create `tests/test_eigenvalue_cache.py`:

```python
"""Tests for eigenvalue cache computation and sharing."""
import numpy as np
import pytest


def test_compute_eigenvalue_cache_symmetric():
    """Eigenvalue cache computes sorted eigenvalues from symmetric matrix."""
    from manylatents.experiment import _compute_eigenvalue_cache

    # Create a simple symmetric PSD matrix
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={None})
    assert (True, None) in cache
    eigs = cache[(True, None)]
    # Eigenvalues of [[2,1],[1,2]] are 1 and 3
    assert len(eigs) == 2
    # Should be sorted descending
    assert eigs[0] >= eigs[1]
    np.testing.assert_allclose(eigs, [3.0, 1.0], atol=1e-10)


def test_compute_eigenvalue_cache_top_k():
    """Eigenvalue cache respects top_k parameter."""
    from manylatents.experiment import _compute_eigenvalue_cache

    A = np.eye(5) * np.arange(1, 6)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={3})
    eigs = cache[(True, 3)]
    assert len(eigs) == 3


def test_eigenvalue_cache_shared_across_metrics():
    """Two metrics requesting same params get same cache entry."""
    from manylatents.experiment import _compute_eigenvalue_cache

    A = np.random.randn(10, 10)
    A = A @ A.T  # Make symmetric PSD

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    cache = _compute_eigenvalue_cache(FakeModule(), top_k_values={None, 5})
    assert (True, None) in cache
    assert (True, 5) in cache
    # Full spectrum should contain top-5
    np.testing.assert_allclose(cache[(True, None)][:5], cache[(True, 5)])
```

**Step 2: Run to verify failure**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_eigenvalue_cache.py -v`
Expected: FAIL — `ImportError: cannot import name '_compute_eigenvalue_cache'`

**Step 3: Add `_compute_eigenvalue_cache` to `experiment.py`**

Add after `_compute_knn_cache` (around line 187):

```python
def _compute_eigenvalue_cache(
    module,
    top_k_values: set[int | None],
) -> dict[tuple[bool, int | None], np.ndarray]:
    """Compute eigenvalues of affinity matrix for sharing across metrics.

    Args:
        module: Fitted LatentModule with affinity_matrix() method.
        top_k_values: Set of top_k values requested by metrics. None means all.

    Returns:
        Dict keyed by (use_symmetric, top_k) -> sorted eigenvalues (descending).
    """
    cache = {}

    try:
        affinity = module.affinity_matrix(use_symmetric=True)
    except (NotImplementedError, AttributeError):
        logger.warning("Module does not expose affinity_matrix; eigenvalue cache empty.")
        return cache

    # Compute full eigenvalue spectrum once
    eigenvalues = np.linalg.eigvalsh(affinity)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    for top_k in top_k_values:
        if top_k is None:
            cache[(True, None)] = eigenvalues_sorted
        else:
            cache[(True, top_k)] = eigenvalues_sorted[:top_k]

    return cache
```

Then, inside `evaluate_embeddings`, after the SVD cache block (around line 259), add:

```python
    # --- Shared eigenvalue computation for spectral metrics ---
    eigenvalue_cache = None
    if module is not None:
        eig_k_values = set()
        for metric_cfg in metric_cfgs.values():
            metric_fn_probe = hydra.utils.instantiate(metric_cfg)
            sig = inspect.signature(metric_fn_probe)
            if "_eigenvalue_cache" in sig.parameters:
                top_k = getattr(metric_cfg, 'top_k', None)
                eig_k_values.add(top_k)
        if eig_k_values:
            logger.info(f"Computing shared eigenvalue cache for top_k values: {eig_k_values}")
            eigenvalue_cache = _compute_eigenvalue_cache(module, eig_k_values)
```

And in the metric dispatch loop, add:

```python
        if "_eigenvalue_cache" in sig.parameters and eigenvalue_cache is not None:
            call_kwargs["_eigenvalue_cache"] = eigenvalue_cache
```

**Step 4: Run tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_eigenvalue_cache.py -v`
Expected: PASS

**Step 5: Verify existing tests still pass**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/ -v --ignore=tests/.venv -x`

**Step 6: Commit**

```bash
git add manylatents/experiment.py tests/test_eigenvalue_cache.py
git commit -m "feat: add eigenvalue cache to evaluation pipeline"
```

---

## Task 8: SpectralGapRatio Metric

**Files:**
- Create: `manylatents/metrics/spectral_gap_ratio.py`
- Create: `manylatents/configs/metrics/module/spectral_gap_ratio.yaml`
- Modify: `manylatents/metrics/__init__.py`
- Test: `tests/test_spectral_metrics.py`

**Step 1: Write failing tests**

Create `tests/test_spectral_metrics.py`:

```python
"""Tests for spectral metrics (SpectralGapRatio, SpectralDecayRate)."""
import numpy as np
import pytest


def test_spectral_gap_ratio_basic():
    """SpectralGapRatio returns lambda_1/lambda_2."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    # Create symmetric PSD matrix with known eigenvalues
    # Eigenvalues: 3, 1
    A = np.array([[2, 1], [1, 2]], dtype=np.float64)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    result = SpectralGapRatio(
        embeddings=np.zeros((2, 2)),
        module=FakeModule(),
    )
    np.testing.assert_allclose(result, 3.0, atol=1e-10)


def test_spectral_gap_ratio_uses_cache():
    """SpectralGapRatio uses _eigenvalue_cache when provided."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    cache = {(True, None): np.array([10.0, 2.0, 1.0])}
    result = SpectralGapRatio(
        embeddings=np.zeros((3, 2)),
        _eigenvalue_cache=cache,
    )
    np.testing.assert_allclose(result, 5.0)


def test_spectral_gap_ratio_no_module_returns_nan():
    """SpectralGapRatio returns nan when no module and no cache."""
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    result = SpectralGapRatio(embeddings=np.zeros((3, 2)))
    assert np.isnan(result)
```

**Step 2: Run to verify failure**

**Step 3: Implement `manylatents/metrics/spectral_gap_ratio.py`**

```python
"""Spectral Gap Ratio metric.

Computes lambda_1 / lambda_2 from the affinity matrix eigenvalue spectrum.
A large gap indicates clear separation between the dominant mode and the rest.
"""
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SpectralGapRatio(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    _eigenvalue_cache: Optional[Dict[Tuple, np.ndarray]] = None,
) -> float:
    """Compute ratio of first to second eigenvalue of the affinity spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused, kept for protocol).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix() method.
        _eigenvalue_cache: Shared eigenvalue cache from evaluate().

    Returns:
        float: lambda_1 / lambda_2, or nan if unavailable.
    """
    eigenvalues = _get_eigenvalues(module, _eigenvalue_cache)
    if eigenvalues is None or len(eigenvalues) < 2:
        return float("nan")

    if eigenvalues[1] == 0:
        return float("inf")

    ratio = float(eigenvalues[0] / eigenvalues[1])
    logger.info(f"SpectralGapRatio: {ratio:.4f}")
    return ratio


def _get_eigenvalues(
    module: Optional[LatentModule],
    cache: Optional[Dict[Tuple, np.ndarray]],
) -> Optional[np.ndarray]:
    """Get eigenvalues from cache or compute from module."""
    if cache is not None:
        # Prefer full spectrum
        for key in [(True, None), (True, 25)]:
            if key in cache:
                return cache[key]
        # Take any cached entry
        if cache:
            return next(iter(cache.values()))

    if module is not None:
        try:
            A = module.affinity_matrix(use_symmetric=True)
            eigs = np.linalg.eigvalsh(A)
            return np.sort(eigs)[::-1]
        except (NotImplementedError, AttributeError):
            warnings.warn(
                f"SpectralGapRatio: {type(module).__name__} does not expose affinity_matrix.",
                RuntimeWarning,
            )

    return None
```

**Step 4: Create Hydra config**

Create `manylatents/configs/metrics/module/spectral_gap_ratio.yaml`:

```yaml
spectral_gap_ratio:
  _target_: manylatents.metrics.spectral_gap_ratio.SpectralGapRatio
  _partial_: True
```

**Step 5: Add to `__init__.py`**

Add to `manylatents/metrics/__init__.py`:

```python
from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
```

And add `"SpectralGapRatio"` to `__all__`.

**Step 6: Run tests, commit**

```bash
git add manylatents/metrics/spectral_gap_ratio.py manylatents/configs/metrics/module/spectral_gap_ratio.yaml manylatents/metrics/__init__.py tests/test_spectral_metrics.py
git commit -m "feat: add SpectralGapRatio metric"
```

---

## Task 9: SpectralDecayRate Metric

**Files:**
- Create: `manylatents/metrics/spectral_decay_rate.py`
- Create: `manylatents/configs/metrics/module/spectral_decay_rate.yaml`
- Modify: `manylatents/metrics/__init__.py`
- Test: Append to `tests/test_spectral_metrics.py`

**Step 1: Add failing tests to `tests/test_spectral_metrics.py`**

```python
def test_spectral_decay_rate_basic():
    """SpectralDecayRate fits exponential decay to eigenvalues."""
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate

    # Create eigenvalues that decay exponentially: exp(-0.5 * i)
    eigs = np.exp(-0.5 * np.arange(20))
    cache = {(True, 20): eigs}

    result = SpectralDecayRate(
        embeddings=np.zeros((20, 2)),
        _eigenvalue_cache=cache,
        top_k=20,
    )
    assert isinstance(result, float)
    assert result > 0  # Decay rate should be positive
    np.testing.assert_allclose(result, 0.5, atol=0.1)
```

**Step 2: Run to verify failure**

**Step 3: Implement `manylatents/metrics/spectral_decay_rate.py`**

```python
"""Spectral Decay Rate metric.

Fits exponential decay lambda_i ~ exp(-rate * i) to the top-k eigenvalues.
Faster decay indicates lower effective dimensionality.
"""
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SpectralDecayRate(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    top_k: int = 20,
    _eigenvalue_cache: Optional[Dict[Tuple, np.ndarray]] = None,
) -> float:
    """Fit exponential decay to the eigenvalue spectrum.

    Args:
        embeddings: Low-dimensional embeddings (unused).
        dataset: Dataset object (unused).
        module: Fitted LatentModule with affinity_matrix().
        top_k: Number of eigenvalues to fit.
        _eigenvalue_cache: Shared eigenvalue cache.

    Returns:
        float: Decay rate (positive = decaying), or nan if unavailable.
    """
    eigenvalues = _get_top_eigenvalues(module, _eigenvalue_cache, top_k)
    if eigenvalues is None or len(eigenvalues) < 3:
        return float("nan")

    # Only use positive eigenvalues for log fit
    pos_mask = eigenvalues > 0
    if pos_mask.sum() < 3:
        return float("nan")

    eigs_pos = eigenvalues[pos_mask]
    log_eigs = np.log(eigs_pos)
    indices = np.arange(len(eigs_pos))

    # Linear fit: log(lambda_i) = -rate * i + intercept
    coeffs = np.polyfit(indices, log_eigs, 1)
    rate = float(-coeffs[0])  # Negate so positive = decaying

    logger.info(f"SpectralDecayRate: {rate:.4f} (top_k={top_k})")
    return rate


def _get_top_eigenvalues(
    module: Optional[LatentModule],
    cache: Optional[Dict[Tuple, np.ndarray]],
    top_k: int,
) -> Optional[np.ndarray]:
    """Get top-k eigenvalues from cache or compute from module."""
    if cache is not None:
        for key in [(True, top_k), (True, None)]:
            if key in cache:
                return cache[key][:top_k]
        if cache:
            eigs = next(iter(cache.values()))
            return eigs[:top_k]

    if module is not None:
        try:
            A = module.affinity_matrix(use_symmetric=True)
            eigs = np.linalg.eigvalsh(A)
            return np.sort(eigs)[::-1][:top_k]
        except (NotImplementedError, AttributeError):
            pass

    return None
```

**Step 4: Create config, update `__init__.py`, run tests, commit**

```bash
git add manylatents/metrics/spectral_decay_rate.py manylatents/configs/metrics/module/spectral_decay_rate.yaml manylatents/metrics/__init__.py tests/test_spectral_metrics.py
git commit -m "feat: add SpectralDecayRate metric"
```

---

## Task 10: SilhouetteScore Metric

**Files:**
- Create: `manylatents/metrics/silhouette.py`
- Create: `manylatents/configs/metrics/embedding/silhouette.yaml`
- Modify: `manylatents/metrics/__init__.py`
- Test: `tests/test_silhouette.py`

**Step 1: Write failing tests**

Create `tests/test_silhouette.py`:

```python
"""Tests for SilhouetteScore metric."""
import numpy as np
import pytest


def test_silhouette_with_labels():
    """SilhouetteScore returns float when labels available."""
    from manylatents.metrics.silhouette import SilhouetteScore

    # Two clear clusters
    emb = np.vstack([np.random.randn(25, 2) + 5, np.random.randn(25, 2) - 5])
    labels = np.array([0] * 25 + [1] * 25)

    class FakeDataset:
        metadata = labels

    result = SilhouetteScore(embeddings=emb, dataset=FakeDataset())
    assert isinstance(result, float)
    assert -1 <= result <= 1
    assert result > 0.5  # Clear clusters → high score


def test_silhouette_no_labels_returns_nan():
    """SilhouetteScore returns nan when no labels available."""
    from manylatents.metrics.silhouette import SilhouetteScore

    class FakeDataset:
        metadata = None

    result = SilhouetteScore(embeddings=np.zeros((10, 2)), dataset=FakeDataset())
    assert np.isnan(result)


def test_silhouette_no_dataset_returns_nan():
    """SilhouetteScore returns nan when no dataset."""
    from manylatents.metrics.silhouette import SilhouetteScore

    result = SilhouetteScore(embeddings=np.zeros((10, 2)))
    assert np.isnan(result)
```

**Step 2: Run to verify failure**

**Step 3: Implement `manylatents/metrics/silhouette.py`**

```python
"""Silhouette Score metric.

Uses torchdr.silhouette_score when available for GPU acceleration,
falls back to sklearn.metrics.silhouette_score.
"""
import logging
import warnings
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def SilhouetteScore(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    metric: str = "euclidean",
) -> float:
    """Compute silhouette coefficient of embedding w.r.t. cluster labels.

    Args:
        embeddings: (n_samples, n_features) embedding array.
        dataset: Dataset with .metadata containing cluster labels.
        module: LatentModule (unused).
        metric: Distance metric for silhouette computation.

    Returns:
        float: Silhouette score in [-1, 1], or nan if labels unavailable.
    """
    labels = _extract_labels(dataset)
    if labels is None:
        warnings.warn("SilhouetteScore: no labels available, returning nan.", RuntimeWarning)
        return float("nan")

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        warnings.warn("SilhouetteScore: fewer than 2 clusters, returning nan.", RuntimeWarning)
        return float("nan")

    try:
        from manylatents.utils.backend import check_torchdr_available

        if check_torchdr_available():
            import torch
            from torchdr import silhouette_score

            X_t = torch.from_numpy(embeddings).float()
            labels_t = torch.from_numpy(labels.astype(np.int64))
            score = silhouette_score(X_t, labels_t, metric=metric)
            result = float(score)
            logger.info(f"SilhouetteScore (torchdr): {result:.4f}")
            return result
    except Exception:
        pass

    from sklearn.metrics import silhouette_score as sk_silhouette

    result = float(sk_silhouette(embeddings, labels, metric=metric))
    logger.info(f"SilhouetteScore (sklearn): {result:.4f}")
    return result


def _extract_labels(dataset: Optional[object]) -> Optional[np.ndarray]:
    """Extract labels from dataset."""
    if dataset is None:
        return None

    labels = getattr(dataset, "metadata", None)
    if labels is None and hasattr(dataset, "get_labels"):
        labels = dataset.get_labels()

    if labels is None:
        return None

    return np.asarray(labels)
```

**Step 4: Create config, update `__init__.py`, run tests, commit**

Config `manylatents/configs/metrics/embedding/silhouette.yaml`:

```yaml
silhouette:
  _target_: manylatents.metrics.silhouette.SilhouetteScore
  _partial_: True
  metric: "euclidean"
```

```bash
git add manylatents/metrics/silhouette.py manylatents/configs/metrics/embedding/silhouette.yaml manylatents/metrics/__init__.py tests/test_silhouette.py
git commit -m "feat: add SilhouetteScore metric with TorchDR GPU acceleration"
```

---

## Task 11: GeodesicDistanceCorrelation Metric

**Files:**
- Create: `manylatents/metrics/geodesic_distance_correlation.py`
- Create: `manylatents/configs/metrics/dataset/geodesic_distance_correlation.yaml`
- Modify: `manylatents/metrics/__init__.py`
- Test: `tests/test_geodesic_correlation.py`

**Step 1: Write failing tests**

Create `tests/test_geodesic_correlation.py`:

```python
"""Tests for GeodesicDistanceCorrelation metric."""
import numpy as np
import pytest


def test_geodesic_correlation_spearman():
    """Spearman correlation between ground truth and embedding distances."""
    from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

    n = 20
    gt_dists = np.random.rand(n, n)
    gt_dists = (gt_dists + gt_dists.T) / 2
    np.fill_diagonal(gt_dists, 0)

    class FakeDataset:
        metadata = None
        def get_gt_dists(self):
            return gt_dists

    emb = np.random.randn(n, 2)
    result = GeodesicDistanceCorrelation(
        embeddings=emb, dataset=FakeDataset(), correlation_type="spearman"
    )
    assert isinstance(result, float)
    assert -1 <= result <= 1


def test_geodesic_correlation_no_gt_returns_nan():
    """Returns nan when dataset has no get_gt_dists."""
    from manylatents.metrics.geodesic_distance_correlation import GeodesicDistanceCorrelation

    class FakeDataset:
        metadata = None

    result = GeodesicDistanceCorrelation(embeddings=np.zeros((10, 2)), dataset=FakeDataset())
    assert np.isnan(result)
```

**Step 2: Run to verify failure**

**Step 3: Implement `manylatents/metrics/geodesic_distance_correlation.py`**

```python
"""Geodesic Distance Correlation metric.

Correlation between ground truth geodesic distances and embedding pairwise distances.
Supports Spearman and Kendall tau correlation types.
"""
import logging
import warnings
from typing import Optional

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import pairwise_distances

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


def GeodesicDistanceCorrelation(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    correlation_type: str = "spearman",
) -> float:
    """Compute correlation between geodesic and embedding pairwise distances.

    Args:
        embeddings: (n_samples, n_features) embedding.
        dataset: Dataset with get_gt_dists() method.
        module: LatentModule (unused).
        correlation_type: "spearman" or "kendall".

    Returns:
        float: Correlation coefficient, or nan if ground truth unavailable.
    """
    if dataset is None or not hasattr(dataset, "get_gt_dists") or not callable(dataset.get_gt_dists):
        warnings.warn("GeodesicDistanceCorrelation: no get_gt_dists() available.", RuntimeWarning)
        return float("nan")

    try:
        gt_dists = dataset.get_gt_dists()
    except Exception:
        return float("nan")

    if gt_dists is None:
        return float("nan")

    emb_dists = pairwise_distances(embeddings, metric="euclidean")

    # Extract upper triangle (avoid diagonal and duplicates)
    triu_idx = np.triu_indices_from(gt_dists, k=1)
    gt_flat = gt_dists[triu_idx]
    emb_flat = emb_dists[triu_idx]

    if correlation_type == "spearman":
        corr, _ = spearmanr(gt_flat, emb_flat)
    elif correlation_type == "kendall":
        corr, _ = kendalltau(gt_flat, emb_flat)
    else:
        raise ValueError(f"Unknown correlation_type: {correlation_type}")

    result = float(corr)
    logger.info(f"GeodesicDistanceCorrelation ({correlation_type}): {result:.4f}")
    return result
```

**Step 4: Create config, update `__init__.py`, run tests, commit**

```bash
git add manylatents/metrics/geodesic_distance_correlation.py manylatents/configs/metrics/dataset/geodesic_distance_correlation.yaml manylatents/metrics/__init__.py tests/test_geodesic_correlation.py
git commit -m "feat: add GeodesicDistanceCorrelation metric"
```

---

## Task 12: DatasetTopologyDescriptor Metric

**Files:**
- Create: `manylatents/metrics/dataset_topology_descriptor.py`
- Create: `manylatents/configs/metrics/module/dataset_topology_descriptor.yaml`
- Modify: `manylatents/metrics/__init__.py`
- Test: `tests/test_topology_descriptor.py`

**Step 1: Write failing tests**

Create `tests/test_topology_descriptor.py`:

```python
"""Tests for DatasetTopologyDescriptor metric."""
import numpy as np
import pytest


def test_topology_descriptor_returns_dict():
    """DatasetTopologyDescriptor returns dict with expected keys."""
    from manylatents.metrics.dataset_topology_descriptor import DatasetTopologyDescriptor

    A = np.eye(5) * np.arange(1, 6)
    cache = {(True, None): np.sort(np.diag(A))[::-1]}

    class FakeDataset:
        metadata = np.arange(5)
        def get_gt_dists(self): return np.eye(5)

    class FakeModule:
        backend = None
        def affinity_matrix(self, use_symmetric=False):
            return A

    result = DatasetTopologyDescriptor(
        embeddings=np.zeros((5, 2)),
        dataset=FakeDataset(),
        module=FakeModule(),
        _eigenvalue_cache=cache,
    )
    assert isinstance(result, dict)
    assert "spectral_gap" in result
    assert "gt_type" in result
```

**Step 2-5: Implement, config, test, commit**

```bash
git commit -m "feat: add DatasetTopologyDescriptor metric"
```

---

## Task 13: AffinityModule

**Files:**
- Create: `manylatents/algorithms/latent/affinity.py`
- Create: `manylatents/configs/algorithms/latent/affinity.yaml`
- Modify: `manylatents/algorithms/latent/__init__.py`
- Test: `tests/test_affinity_module.py`

**Step 1: Write failing tests**

Create `tests/test_affinity_module.py`:

```python
"""Tests for AffinityModule."""
import numpy as np
import pytest
import torch


def test_affinity_module_fit_transform():
    """AffinityModule.fit() builds affinity, transform() returns input."""
    from manylatents.algorithms.latent.affinity import AffinityModule

    m = AffinityModule(n_components=2, knn=5)
    x = torch.randn(30, 10)
    m.fit(x)
    assert m._is_fitted
    result = m.transform(x)
    # transform is identity
    np.testing.assert_allclose(result.numpy(), x.numpy())


def test_affinity_module_kernel_matrix():
    """AffinityModule.kernel_matrix() returns NxN matrix."""
    from manylatents.algorithms.latent.affinity import AffinityModule

    m = AffinityModule(n_components=2, knn=5)
    x = torch.randn(30, 10)
    m.fit(x)
    K = m.kernel_matrix()
    assert K.shape == (30, 30)
    # Kernel should be symmetric
    np.testing.assert_allclose(K, K.T, atol=1e-10)


def test_affinity_module_symmetric_affinity():
    """AffinityModule.affinity_matrix(use_symmetric=True) returns symmetric matrix."""
    from manylatents.algorithms.latent.affinity import AffinityModule

    m = AffinityModule(n_components=2, knn=5)
    x = torch.randn(30, 10)
    m.fit(x)
    A = m.affinity_matrix(use_symmetric=True)
    assert A.shape == (30, 30)
    np.testing.assert_allclose(A, A.T, atol=1e-10)
```

**Step 2: Run to verify failure**

**Step 3: Implement `manylatents/algorithms/latent/affinity.py`**

```python
"""Standalone Affinity Module.

Computes and exposes affinity/kernel matrices without dimensionality reduction.
Useful for pre-embedding spectral analysis.
"""
import logging
from typing import Optional

import graphtools
import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule
from ...utils.kernel_utils import symmetric_diffusion_operator

logger = logging.getLogger(__name__)


class AffinityModule(LatentModule):
    """LatentModule that computes affinity matrices without DR.

    transform() returns input unchanged (identity). The primary purpose is
    exposing kernel_matrix() and affinity_matrix() for spectral analysis.
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 15,
        alpha: float = 1.0,
        symmetric: bool = True,
        metric: str = "euclidean",
        decay: int = 40,
        backend: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components, backend=backend, device=device, **kwargs
        )
        self.knn = knn
        self.alpha = alpha
        self.symmetric = symmetric
        self.metric = metric
        self.decay = decay
        self._kernel = None

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        x_np = x.detach().cpu().numpy()
        G = graphtools.Graph(
            x_np,
            knn=self.knn,
            decay=self.decay,
            distance=self.metric,
            n_jobs=-1,
            verbose=0,
        )
        self._kernel = np.asarray(G.kernel.todense())
        self._is_fitted = True
        logger.info(f"AffinityModule fitted: kernel shape {self._kernel.shape}")

    def transform(self, x: Tensor) -> Tensor:
        return x

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("AffinityModule not fitted. Call fit() first.")
        K = self._kernel.copy()
        if ignore_diagonal:
            np.fill_diagonal(K, 0)
        return K

    def affinity_matrix(
        self, ignore_diagonal: bool = False, use_symmetric: bool = False
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("AffinityModule not fitted. Call fit() first.")
        if use_symmetric:
            return symmetric_diffusion_operator(self._kernel, self.alpha)
        K = self._kernel.copy()
        if ignore_diagonal:
            np.fill_diagonal(K, 0)
        # Row-stochastic normalization
        row_sums = K.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return K / row_sums
```

**Step 4: Create config and register**

Create `manylatents/configs/algorithms/latent/affinity.yaml`:

```yaml
_target_: manylatents.algorithms.latent.affinity.AffinityModule
n_components: 2
knn: 15
alpha: 1.0
symmetric: true
metric: "euclidean"
decay: 40
backend: null
device: null
```

Add to `manylatents/algorithms/latent/__init__.py`:

```python
from .affinity import AffinityModule
```

And add `"AffinityModule"` to `__all__`.

**Step 5: Run tests, commit**

```bash
git add manylatents/algorithms/latent/affinity.py manylatents/configs/algorithms/latent/affinity.yaml manylatents/algorithms/latent/__init__.py tests/test_affinity_module.py
git commit -m "feat: add standalone AffinityModule for spectral analysis"
```

---

## Task 14: Sweep Configs

**Files:**
- Create: `manylatents/configs/sweep/dataset_algorithm_grid.yaml`
- Create: `manylatents/configs/sweep/umap_parameter_sensitivity.yaml`
- Create: `manylatents/configs/sweep/phate_parameter_sensitivity.yaml`
- Create: `manylatents/configs/sweep/backend_comparison.yaml`
- Test: `tests/test_sweep_configs.py`

**Step 1: Create sweep configs**

`manylatents/configs/sweep/dataset_algorithm_grid.yaml`:

```yaml
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

`manylatents/configs/sweep/umap_parameter_sensitivity.yaml`:

```yaml
defaults:
  - override /algorithms/latent: umap
  - override /data: swissroll
  - override /metrics/embedding: [trustworthiness, continuity, knn_preservation]

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      algorithms.latent.n_neighbors: 5,10,15,30,50
      algorithms.latent.min_dist: 0.01,0.1,0.5,1.0
      seed: 42,43,44
```

`manylatents/configs/sweep/phate_parameter_sensitivity.yaml`:

```yaml
defaults:
  - override /algorithms/latent: phate
  - override /data: swissroll
  - override /metrics/embedding: [trustworthiness, continuity, knn_preservation]

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      algorithms.latent.knn: 3,5,10,15,30
      algorithms.latent.t: 5,15,50,100
      seed: 42,43,44
```

`manylatents/configs/sweep/backend_comparison.yaml`:

```yaml
defaults:
  - override /data: swissroll
  - override /metrics/embedding: [trustworthiness, continuity, knn_preservation]
  - override /metrics/module: [affinity_spectrum]

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      algorithms/latent: umap,umap_torchdr,tsne,tsne_torchdr
      seed: 42,43,44
```

**Step 2: Write validation test**

Create `tests/test_sweep_configs.py`:

```python
"""Tests for sweep config validity."""
import pytest
from pathlib import Path

SWEEP_DIR = Path(__file__).parent.parent / "manylatents" / "configs" / "sweep"


def test_sweep_configs_exist():
    """All sweep config files exist."""
    expected = [
        "dataset_algorithm_grid.yaml",
        "umap_parameter_sensitivity.yaml",
        "phate_parameter_sensitivity.yaml",
        "backend_comparison.yaml",
    ]
    for name in expected:
        assert (SWEEP_DIR / name).exists(), f"Missing sweep config: {name}"


def test_sweep_configs_are_valid_yaml():
    """All sweep configs parse as valid YAML."""
    import yaml

    for path in SWEEP_DIR.glob("*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None, f"Empty config: {path.name}"
```

**Step 3: Run tests, commit**

```bash
git add manylatents/configs/sweep/dataset_algorithm_grid.yaml manylatents/configs/sweep/umap_parameter_sensitivity.yaml manylatents/configs/sweep/phate_parameter_sensitivity.yaml manylatents/configs/sweep/backend_comparison.yaml tests/test_sweep_configs.py
git commit -m "feat: add sweep configs for grid, sensitivity, and backend comparison"
```

---

## Task 15: Enhanced merge_results.py

**Files:**
- Modify: `manylatents/utils/merge_results.py`
- Test: `tests/test_merge_results.py`

**Step 1: Write failing tests**

Create `tests/test_merge_results.py`:

```python
"""Tests for merge_results enhancements."""
import numpy as np
import pandas as pd
import pytest


def test_generate_pivot_table():
    """generate_pivot_table produces expected format."""
    from manylatents.utils.merge_results import generate_pivot_table

    df = pd.DataFrame({
        "data": ["swissroll", "swissroll", "torus", "torus"],
        "algorithm": ["umap", "pca", "umap", "pca"],
        "trustworthiness": [0.95, 0.80, 0.90, 0.75],
        "seed": [42, 42, 42, 42],
    })
    pivot = generate_pivot_table(df, metric_cols=["trustworthiness"])
    assert "trustworthiness" in pivot.columns
    assert len(pivot) == 4  # 2 datasets x 2 algorithms


def test_parameter_sensitivity_summary():
    """parameter_sensitivity_summary computes mean/std across seeds."""
    from manylatents.utils.merge_results import parameter_sensitivity_summary

    df = pd.DataFrame({
        "n_neighbors": [5, 5, 10, 10],
        "min_dist": [0.1, 0.1, 0.1, 0.1],
        "trustworthiness": [0.90, 0.92, 0.88, 0.86],
        "seed": [42, 43, 42, 43],
    })
    summary = parameter_sensitivity_summary(
        df,
        param_cols=["n_neighbors", "min_dist"],
        metric_cols=["trustworthiness"],
    )
    assert "trustworthiness_mean" in summary.columns
    assert "trustworthiness_std" in summary.columns
    assert len(summary) == 2  # 2 unique (n_neighbors, min_dist) combos
```

**Step 2: Run to verify failure**

**Step 3: Add functions to `merge_results.py`**

Add after `load_all_runs_from_sweeps`:

```python
def generate_pivot_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    index_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate pivot table with metrics as columns.

    Args:
        df: DataFrame with columns for data, algorithm, metrics, seed.
        metric_cols: Metric column names to include.
        index_cols: Columns to use as index. Defaults to ["data", "algorithm"].

    Returns:
        DataFrame with (data, algorithm) rows and metric columns.
    """
    if index_cols is None:
        index_cols = [c for c in ["data", "algorithm"] if c in df.columns]
    return df.groupby(index_cols)[metric_cols].mean().reset_index()


def parameter_sensitivity_summary(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Compute mean and std of metrics across seeds for each parameter combo.

    Args:
        df: DataFrame with parameter and metric columns.
        param_cols: Parameter column names to group by.
        metric_cols: Metric column names to summarize.

    Returns:
        DataFrame with mean and std columns for each metric.
    """
    agg_dict = {}
    for m in metric_cols:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")
    return df.groupby(param_cols).agg(**agg_dict).reset_index()
```

Also update the CLI `--pivot` flag in `main()`:

```python
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Generate pivot table (mean across seeds)"
    )
```

**Step 4: Run tests, commit**

```bash
git add manylatents/utils/merge_results.py tests/test_merge_results.py
git commit -m "feat: add pivot table and sensitivity summary to merge_results"
```

---

## Task 16: Capability Logging in Evaluation Pipeline

**Files:**
- Modify: `manylatents/experiment.py:196-230`

**Step 1: Add capability logging to `evaluate_embeddings`**

In `evaluate_embeddings`, after `module = kwargs.get("module", None)` (line 228), add:

```python
    # Log dataset capabilities
    from manylatents.data.capabilities import log_capabilities
    log_capabilities(ds_sub)
```

**Step 2: Run existing tests to verify no regression**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/ -v --ignore=tests/.venv -x`

**Step 3: Commit**

```bash
git add manylatents/experiment.py
git commit -m "feat: log dataset capabilities during evaluation"
```

---

## Task 17: MetricAgreement (Post-hoc Analysis)

**Files:**
- Create: `manylatents/metrics/metric_agreement.py`
- Test: `tests/test_metric_agreement.py`

**Step 1: Write failing tests**

```python
"""Tests for MetricAgreement post-hoc analysis."""
import numpy as np
import pandas as pd
import pytest


def test_metric_agreement_returns_correlation_matrix():
    """MetricAgreement returns pairwise Spearman correlation matrix."""
    from manylatents.metrics.metric_agreement import MetricAgreement

    df = pd.DataFrame({
        "trustworthiness": [0.95, 0.80, 0.70, 0.60],
        "continuity": [0.90, 0.75, 0.65, 0.55],
        "knn_preservation": [0.50, 0.85, 0.90, 0.30],
    })
    result = MetricAgreement(df, metric_cols=["trustworthiness", "continuity", "knn_preservation"])
    assert result.shape == (3, 3)
    # Diagonal should be 1.0
    np.testing.assert_allclose(np.diag(result.values), 1.0)
```

**Step 2-4: Implement, test, commit**

```python
"""Metric Agreement analysis.

Computes pairwise Spearman rank correlation between metrics across runs.
This is a post-hoc analysis tool, not a standard embedding metric.
"""
import pandas as pd
from scipy.stats import spearmanr


def MetricAgreement(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between metrics across runs.

    Args:
        df: DataFrame with metric columns.
        metric_cols: Column names of metrics to compare.

    Returns:
        DataFrame: Symmetric correlation matrix.
    """
    subset = df[metric_cols].dropna()
    corr_matrix, _ = spearmanr(subset.values)
    if len(metric_cols) == 2:
        corr_matrix = [[1.0, corr_matrix], [corr_matrix, 1.0]]
    return pd.DataFrame(corr_matrix, index=metric_cols, columns=metric_cols)
```

```bash
git add manylatents/metrics/metric_agreement.py manylatents/metrics/__init__.py tests/test_metric_agreement.py
git commit -m "feat: add MetricAgreement post-hoc analysis"
```

---

## Task 18: Full Integration Test

**Files:**
- Test: `tests/test_gpu_metric_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for GPU metric infrastructure."""
import numpy as np
import pytest
import torch


def test_full_pipeline_cpu():
    """End-to-end: UMAPModule + spectral metrics + silhouette on SwissRoll."""
    from manylatents.algorithms.latent.umap import UMAPModule
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate
    from manylatents.metrics.silhouette import SilhouetteScore
    from manylatents.data.capabilities import get_capabilities
    from manylatents.experiment import _compute_eigenvalue_cache

    # Generate simple data
    x = torch.randn(100, 10)

    # Fit UMAP
    m = UMAPModule(n_components=2, random_state=42, n_neighbors=10, n_epochs=50)
    emb = m.fit_transform(x)
    emb_np = emb.numpy()

    # Compute eigenvalue cache
    cache = _compute_eigenvalue_cache(m, top_k_values={None, 20})

    # Run spectral metrics
    gap = SpectralGapRatio(emb_np, module=m, _eigenvalue_cache=cache)
    assert isinstance(gap, float)
    assert not np.isnan(gap)

    decay = SpectralDecayRate(emb_np, module=m, _eigenvalue_cache=cache, top_k=20)
    assert isinstance(decay, float)

    # Silhouette with fake labels
    class FakeDS:
        metadata = np.random.randint(0, 3, size=100)

    sil = SilhouetteScore(emb_np, dataset=FakeDS())
    assert isinstance(sil, float)
    assert -1 <= sil <= 1


def test_affinity_module_with_metrics():
    """AffinityModule produces valid affinity for spectral metrics."""
    from manylatents.algorithms.latent.affinity import AffinityModule
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio

    m = AffinityModule(knn=5)
    x = torch.randn(50, 10)
    m.fit(x)

    gap = SpectralGapRatio(embeddings=x.numpy(), module=m)
    assert isinstance(gap, float)
    assert not np.isnan(gap)
    assert gap > 0


def test_dataset_capabilities_on_real_datasets():
    """get_capabilities works on actual synthetic datasets."""
    from manylatents.data.capabilities import get_capabilities
    from manylatents.data.synthetic_dataset import SwissRoll

    ds = SwissRoll()
    caps = get_capabilities(ds)
    assert caps["gt_dists"] is False or caps["gt_dists"] is True
    assert caps["labels"] is True
    assert "gt_type" in caps
```

**Step 2: Run integration tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/test_gpu_metric_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_gpu_metric_integration.py
git commit -m "test: add integration tests for GPU metric infrastructure"
```

---

## Task 19: Final Verification and Cleanup

**Step 1: Run all tests**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m pytest tests/ -v --ignore=tests/.venv -x`
Expected: ALL PASS

**Step 2: Verify imports work**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -c "from manylatents.utils.backend import check_torchdr_available; from manylatents.data.capabilities import get_capabilities; from manylatents.metrics import SpectralGapRatio, SpectralDecayRate, SilhouetteScore; from manylatents.algorithms.latent import AffinityModule; print('All imports OK')"`

**Step 3: Verify Hydra config resolution**

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m manylatents.main --cfg job algorithms/latent=umap 2>&1 | head -20`
Expected: Config should show `backend: null, device: null`

Run: `cd /network/scratch/c/cesar.valdez/lrw/manylatents && python -m manylatents.main --cfg job algorithms/latent=umap_torchdr 2>&1 | head -20`
Expected: Config should show `backend: torchdr, device: auto`

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for GPU metric infrastructure"
```

---

## Summary: Task Dependency Graph

```
Task 1 (Dependencies) ─────┐
                            ├── Task 3 (Base class) ──┬── Task 4 (UMAP backend)
Task 2 (Capabilities) ─────┤                         ├── Task 5 (PHATE backend)
                            │                         ├── Task 6 (TSNE backend)
                            │                         ├── Task 13 (AffinityModule)
                            │                         └── Task 7 (Eigenvalue cache)
                            │                                      │
                            ├── Task 8 (SpectralGapRatio) ─────────┤
                            ├── Task 9 (SpectralDecayRate) ────────┤
                            ├── Task 10 (SilhouetteScore)          │
                            ├── Task 11 (GeodesicCorrelation)      │
                            ├── Task 12 (TopologyDescriptor) ──────┘
                            ├── Task 14 (Sweep configs)
                            ├── Task 15 (Enhanced merge_results)
                            ├── Task 16 (Capability logging)
                            ├── Task 17 (MetricAgreement)
                            ├── Task 18 (Integration tests)
                            └── Task 19 (Final verification)
```

**Parallelizable groups:**
- Tasks 1 + 2 (independent foundations)
- Tasks 4 + 5 + 6 (independent backend implementations, after Task 3)
- Tasks 8 + 9 + 10 + 11 + 12 (independent metrics, after Task 7)
- Tasks 14 + 15 + 16 + 17 (independent infrastructure)
