# HF Representation Audit Infrastructure - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build infrastructure for auditing neural network representations during training, computing diffusion operators from activations, and merging representations across models.

**Architecture:** Lightning callbacks extract activations via PyTorch forward hooks at configurable triggers. Activations flow through DiffusionGauge (reusing manylatents' symmetric_diffusion_operator) to produce diffusion operators. Extended MergingModule supports operator-level fusion (weighted interpolation, Frobenius mean, OT barycenters). Trajectories logged to wandb, parallel runs via shop/SLURM.

**Tech Stack:** PyTorch Lightning, HuggingFace Transformers, manylatents (PHATE, diffusion operators), wandb, shop hydra launchers

**Reference:** See `docs/designs/hf-representation-audit-architecture.md` for diagrams.

---

## Phase 1: Core Extraction Infrastructure

### Task 1: LayerSpec dataclass

**Files:**
- Create: `manylatents/lightning/hooks.py`
- Test: `manylatents/lightning/tests/test_hooks.py`

**Step 1: Write the failing test**

```python
# manylatents/lightning/tests/test_hooks.py
import pytest
from manylatents.lightning.hooks import LayerSpec


def test_layer_spec_defaults():
    spec = LayerSpec(path="model.layers[-1]")
    assert spec.path == "model.layers[-1]"
    assert spec.extraction_point == "output"
    assert spec.reduce == "mean"


def test_layer_spec_custom():
    spec = LayerSpec(
        path="model.layers[12].self_attn",
        extraction_point="hidden_states",
        reduce="last_token",
    )
    assert spec.extraction_point == "hidden_states"
    assert spec.reduce == "last_token"


def test_layer_spec_invalid_reduce():
    with pytest.raises(ValueError, match="reduce must be one of"):
        LayerSpec(path="model.layers[-1]", reduce="invalid")
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/tests/test_hooks.py::test_layer_spec_defaults -v`
Expected: FAIL with "No module named 'manylatents.lightning.hooks'"

**Step 3: Write minimal implementation**

```python
# manylatents/lightning/hooks.py
"""Activation extraction via PyTorch forward hooks."""
from dataclasses import dataclass, field
from typing import Literal

VALID_REDUCE = ("mean", "last_token", "cls", "first_token", "all", "none")
VALID_EXTRACTION_POINTS = ("output", "input", "hidden_states")


@dataclass
class LayerSpec:
    """Specification for which layer to extract activations from.

    Attributes:
        path: Dot-separated path to layer, e.g., "model.layers[-1]"
        extraction_point: What to extract - "output", "input", or "hidden_states"
        reduce: How to reduce sequence dimension:
            - "mean": Average over sequence
            - "last_token": Take last token
            - "cls" / "first_token": Take first token (CLS)
            - "all": Keep full sequence (N, seq_len, dim)
            - "none": No reduction, return raw
    """
    path: str
    extraction_point: Literal["output", "input", "hidden_states"] = "output"
    reduce: Literal["mean", "last_token", "cls", "first_token", "all", "none"] = "mean"

    def __post_init__(self):
        if self.reduce not in VALID_REDUCE:
            raise ValueError(
                f"reduce must be one of {VALID_REDUCE}, got '{self.reduce}'"
            )
        if self.extraction_point not in VALID_EXTRACTION_POINTS:
            raise ValueError(
                f"extraction_point must be one of {VALID_EXTRACTION_POINTS}, "
                f"got '{self.extraction_point}'"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/tests/test_hooks.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/hooks.py manylatents/lightning/tests/test_hooks.py
git commit -m "feat(lightning): add LayerSpec dataclass for activation extraction"
```

---

### Task 2: resolve_layer utility

**Files:**
- Modify: `manylatents/lightning/hooks.py`
- Test: `manylatents/lightning/tests/test_hooks.py`

**Step 1: Write the failing test**

```python
# Add to manylatents/lightning/tests/test_hooks.py
import torch.nn as nn
from manylatents.lightning.hooks import resolve_layer


class MockTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(64, 64)
        self.mlp = nn.Linear(64, 64)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.layers = nn.ModuleList([MockTransformerBlock() for _ in range(4)])
        self.lm_head = nn.Linear(64, 100)


def test_resolve_layer_simple():
    model = MockModel()
    layer = resolve_layer(model, "lm_head")
    assert layer is model.lm_head


def test_resolve_layer_nested():
    model = MockModel()
    layer = resolve_layer(model, "layers[2].self_attn")
    assert layer is model.layers[2].self_attn


def test_resolve_layer_negative_index():
    model = MockModel()
    layer = resolve_layer(model, "layers[-1]")
    assert layer is model.layers[-1]


def test_resolve_layer_invalid():
    model = MockModel()
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
        resolve_layer(model, "nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/tests/test_hooks.py::test_resolve_layer_simple -v`
Expected: FAIL with "cannot import name 'resolve_layer'"

**Step 3: Write minimal implementation**

```python
# Add to manylatents/lightning/hooks.py
import re
from typing import Any
import torch.nn as nn


def resolve_layer(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated path to a layer in a model.

    Supports:
    - Dot notation: "layers.0.self_attn"
    - Index notation: "layers[0].self_attn"
    - Negative indices: "layers[-1]"

    Args:
        model: The model to traverse
        path: Dot-separated path with optional bracket indices

    Returns:
        The resolved layer

    Raises:
        AttributeError: If path component doesn't exist
        IndexError: If index is out of bounds
    """
    current: Any = model

    # Split on dots, but preserve bracket notation
    # "layers[-1].self_attn" -> ["layers[-1]", "self_attn"]
    parts = re.split(r'\.(?![^\[]*\])', path)

    for part in parts:
        # Check for index notation: "layers[0]" or "layers[-1]"
        match = re.match(r'(\w+)\[(-?\d+)\]', part)
        if match:
            attr_name, idx = match.groups()
            current = getattr(current, attr_name)
            current = current[int(idx)]
        else:
            if not hasattr(current, part):
                raise AttributeError(
                    f"'{type(current).__name__}' has no attribute '{part}'"
                )
            current = getattr(current, part)

    return current
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/tests/test_hooks.py::test_resolve_layer_simple manylatents/lightning/tests/test_hooks.py::test_resolve_layer_nested manylatents/lightning/tests/test_hooks.py::test_resolve_layer_negative_index manylatents/lightning/tests/test_hooks.py::test_resolve_layer_invalid -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/hooks.py manylatents/lightning/tests/test_hooks.py
git commit -m "feat(lightning): add resolve_layer for dot-path layer access"
```

---

### Task 3: ActivationExtractor core

**Files:**
- Modify: `manylatents/lightning/hooks.py`
- Test: `manylatents/lightning/tests/test_hooks.py`

**Step 1: Write the failing test**

```python
# Add to manylatents/lightning/tests/test_hooks.py
import torch
from manylatents.lightning.hooks import ActivationExtractor, LayerSpec


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


def test_activation_extractor_single_layer():
    model = SimpleModel()
    spec = LayerSpec(path="layer1", reduce="none")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert "layer1" in activations
    assert activations["layer1"].shape == (4, 20)


def test_activation_extractor_multiple_layers():
    model = SimpleModel()
    specs = [
        LayerSpec(path="layer1", reduce="none"),
        LayerSpec(path="layer2", reduce="none"),
    ]
    extractor = ActivationExtractor(specs)

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert len(activations) == 2
    assert activations["layer1"].shape == (4, 20)
    assert activations["layer2"].shape == (4, 5)


def test_activation_extractor_clears_after_get():
    model = SimpleModel()
    spec = LayerSpec(path="layer1", reduce="none")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 10)

    with extractor.capture(model):
        _ = model(x)

    _ = extractor.get_activations()

    # Second call should return empty (already cleared)
    activations2 = extractor.get_activations()
    assert len(activations2) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/tests/test_hooks.py::test_activation_extractor_single_layer -v`
Expected: FAIL with "cannot import name 'ActivationExtractor'"

**Step 3: Write minimal implementation**

```python
# Add to manylatents/lightning/hooks.py
from contextlib import contextmanager
from typing import Dict, List
import torch
from torch import Tensor


class ActivationExtractor:
    """Extract activations from specified layers using forward hooks.

    Usage:
        extractor = ActivationExtractor([LayerSpec("model.layers[-1]")])
        with extractor.capture(model):
            model(inputs)
        activations = extractor.get_activations()
    """

    def __init__(self, layer_specs: List[LayerSpec]):
        self.layer_specs = layer_specs
        self._activations: Dict[str, List[Tensor]] = {}
        self._handles: List = []

    def _make_hook(self, spec: LayerSpec):
        """Create a forward hook for the given spec."""
        def hook(module, input, output):
            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Reduce sequence dimension if needed
            tensor = self._reduce(tensor, spec.reduce)

            if spec.path not in self._activations:
                self._activations[spec.path] = []
            self._activations[spec.path].append(tensor.detach())

        return hook

    def _reduce(self, tensor: Tensor, method: str) -> Tensor:
        """Reduce sequence dimension."""
        if method == "none" or tensor.dim() < 3:
            return tensor

        # Assume shape is (batch, seq_len, dim)
        if method == "mean":
            return tensor.mean(dim=1)
        elif method == "last_token":
            return tensor[:, -1, :]
        elif method in ("cls", "first_token"):
            return tensor[:, 0, :]
        elif method == "all":
            return tensor
        else:
            return tensor

    @contextmanager
    def capture(self, model: nn.Module):
        """Context manager to capture activations during forward pass."""
        # Register hooks
        for spec in self.layer_specs:
            layer = resolve_layer(model, spec.path)
            handle = layer.register_forward_hook(self._make_hook(spec))
            self._handles.append(handle)

        try:
            yield self
        finally:
            # Remove hooks
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    def get_activations(self, clear: bool = True) -> Dict[str, Tensor]:
        """Get captured activations, concatenated over batches.

        Args:
            clear: Whether to clear activations after getting them

        Returns:
            Dict mapping layer path to activation tensor
        """
        result = {}
        for path, tensors in self._activations.items():
            if tensors:
                result[path] = torch.cat(tensors, dim=0)

        if clear:
            self._activations.clear()

        return result

    def clear(self):
        """Clear stored activations."""
        self._activations.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/tests/test_hooks.py::test_activation_extractor_single_layer manylatents/lightning/tests/test_hooks.py::test_activation_extractor_multiple_layers manylatents/lightning/tests/test_hooks.py::test_activation_extractor_clears_after_get -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/hooks.py manylatents/lightning/tests/test_hooks.py
git commit -m "feat(lightning): add ActivationExtractor with forward hooks"
```

---

### Task 4: ActivationExtractor sequence reduction tests

**Files:**
- Modify: `manylatents/lightning/tests/test_hooks.py`

**Step 1: Write additional reduction tests**

```python
# Add to manylatents/lightning/tests/test_hooks.py

class SequenceModel(nn.Module):
    """Model that outputs (batch, seq_len, dim)."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        # x: (batch, seq_len, 10) -> (batch, seq_len, 20)
        return self.layer(x)


def test_activation_extractor_reduce_mean():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="mean")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)  # batch=4, seq_len=8, dim=10

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 20)  # Reduced over seq_len


def test_activation_extractor_reduce_last_token():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="last_token")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 20)


def test_activation_extractor_reduce_all():
    model = SequenceModel()
    spec = LayerSpec(path="layer", reduce="all")
    extractor = ActivationExtractor([spec])

    x = torch.randn(4, 8, 10)

    with extractor.capture(model):
        _ = model(x)

    activations = extractor.get_activations()
    assert activations["layer"].shape == (4, 8, 20)  # Kept full sequence
```

**Step 2: Run tests**

Run: `pytest manylatents/lightning/tests/test_hooks.py -v -k "reduce"`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add manylatents/lightning/tests/test_hooks.py
git commit -m "test(lightning): add ActivationExtractor reduction tests"
```

---

## Phase 2: DiffusionGauge

### Task 5: DiffusionGauge core

**Files:**
- Create: `manylatents/gauge/__init__.py`
- Create: `manylatents/gauge/diffusion.py`
- Test: `manylatents/gauge/tests/test_diffusion.py`

**Step 1: Write the failing test**

```python
# manylatents/gauge/tests/test_diffusion.py
import pytest
import numpy as np
import torch
from manylatents.gauge.diffusion import DiffusionGauge


def test_diffusion_gauge_basic():
    """Gauge should produce a row-stochastic matrix."""
    gauge = DiffusionGauge()
    activations = torch.randn(100, 64)

    diff_op = gauge(activations)

    assert diff_op.shape == (100, 100)
    # Row-stochastic: rows sum to 1
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_gauge_symmetric():
    """Symmetric mode should produce symmetric matrix."""
    gauge = DiffusionGauge(symmetric=True)
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)
    np.testing.assert_allclose(diff_op, diff_op.T, rtol=1e-5)


def test_diffusion_gauge_deterministic():
    """Same input should produce same output."""
    gauge = DiffusionGauge()
    activations = torch.randn(30, 16)

    diff_op1 = gauge(activations)
    diff_op2 = gauge(activations)

    np.testing.assert_allclose(diff_op1, diff_op2)
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/gauge/tests/test_diffusion.py::test_diffusion_gauge_basic -v`
Expected: FAIL with "No module named 'manylatents.gauge'"

**Step 3: Create package structure**

```python
# manylatents/gauge/__init__.py
"""Gauge modules for computing geometric operators from representations."""
from manylatents.gauge.diffusion import DiffusionGauge

__all__ = ["DiffusionGauge"]
```

```python
# manylatents/gauge/tests/__init__.py
# Empty init for test discovery
```

**Step 4: Write minimal implementation**

```python
# manylatents/gauge/diffusion.py
"""Compute diffusion operators from activation tensors."""
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.distance import pdist, squareform

# Reuse existing manylatents machinery
from manylatents.algorithms.latent.phate import symmetric_diffusion_operator


@dataclass
class DiffusionGauge:
    """Compute diffusion operator from activation tensors.

    Pipeline:
        activations (N, D) -> pairwise distances -> Gaussian kernel ->
        affinity matrix -> diffusion operator

    Attributes:
        knn: Number of neighbors for adaptive bandwidth. If None, uses global bandwidth.
        alpha: Diffusion normalization parameter (0=graph Laplacian, 1=Laplace-Beltrami)
        symmetric: If True, return symmetric operator D^{-1/2} K D^{-1/2}
        metric: Distance metric for pairwise computation
    """
    knn: Optional[int] = 15
    alpha: float = 1.0
    symmetric: bool = False
    metric: str = "euclidean"

    def __call__(self, activations: Tensor) -> np.ndarray:
        """Compute diffusion operator from activations.

        Args:
            activations: Tensor of shape (N, D) - N samples, D dimensions

        Returns:
            Diffusion operator of shape (N, N)
        """
        if isinstance(activations, Tensor):
            activations = activations.detach().cpu().numpy()

        # Compute pairwise distances
        distances = squareform(pdist(activations, metric=self.metric))

        # Compute adaptive bandwidth (local scaling)
        if self.knn is not None:
            # k-th nearest neighbor distance for each point
            sorted_dists = np.sort(distances, axis=1)
            sigma = sorted_dists[:, min(self.knn, distances.shape[0] - 1)]
            sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero
            # Adaptive Gaussian kernel
            kernel = np.exp(-distances**2 / (sigma[:, None] * sigma[None, :]))
        else:
            # Global bandwidth (median heuristic)
            sigma = np.median(distances[distances > 0])
            kernel = np.exp(-distances**2 / (2 * sigma**2))

        # Zero out diagonal for cleaner diffusion
        np.fill_diagonal(kernel, 0)

        if self.symmetric:
            return symmetric_diffusion_operator(kernel, alpha=self.alpha)
        else:
            # Row-stochastic normalization
            row_sums = kernel.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            return kernel / row_sums
```

**Step 5: Run test to verify it passes**

Run: `pytest manylatents/gauge/tests/test_diffusion.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add manylatents/gauge/
git commit -m "feat(gauge): add DiffusionGauge for activation → diffusion operator"
```

---

### Task 6: DiffusionGauge kernel options

**Files:**
- Modify: `manylatents/gauge/diffusion.py`
- Modify: `manylatents/gauge/tests/test_diffusion.py`

**Step 1: Write failing tests for kernel options**

```python
# Add to manylatents/gauge/tests/test_diffusion.py

def test_diffusion_gauge_cosine_metric():
    """Should work with cosine distance."""
    gauge = DiffusionGauge(metric="cosine")
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)
    row_sums = diff_op.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_gauge_no_knn():
    """Global bandwidth when knn=None."""
    gauge = DiffusionGauge(knn=None)
    activations = torch.randn(50, 32)

    diff_op = gauge(activations)

    assert diff_op.shape == (50, 50)


def test_diffusion_gauge_alpha_zero():
    """alpha=0 gives graph Laplacian normalization."""
    gauge = DiffusionGauge(alpha=0.0, symmetric=True)
    activations = torch.randn(30, 16)

    diff_op = gauge(activations)

    # Still symmetric
    np.testing.assert_allclose(diff_op, diff_op.T, rtol=1e-5)
```

**Step 2: Run tests**

Run: `pytest manylatents/gauge/tests/test_diffusion.py -v`
Expected: PASS (6 tests)

**Step 3: Commit**

```bash
git add manylatents/gauge/tests/test_diffusion.py
git commit -m "test(gauge): add DiffusionGauge kernel option tests"
```

---

## Phase 3: HFTrainerModule

### Task 7: HFTrainerModule skeleton

**Files:**
- Create: `manylatents/lightning/hf_trainer.py`
- Test: `manylatents/lightning/tests/test_hf_trainer.py`

**Step 1: Write the failing test**

```python
# manylatents/lightning/tests/test_hf_trainer.py
import pytest
import torch
from lightning import Trainer
from manylatents.lightning.hf_trainer import HFTrainerModule, HFTrainerConfig


def test_hf_trainer_module_instantiation():
    """Should instantiate with config."""
    config = HFTrainerConfig(
        model_name_or_path="gpt2",
        learning_rate=2e-5,
    )
    module = HFTrainerModule(config)

    assert module.config == config
    assert module.network is None  # Lazy init


def test_hf_trainer_config_defaults():
    """Config should have sensible defaults."""
    config = HFTrainerConfig(model_name_or_path="gpt2")

    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.0
    assert config.warmup_steps == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/tests/test_hf_trainer.py::test_hf_trainer_module_instantiation -v`
Expected: FAIL with "cannot import name 'HFTrainerModule'"

**Step 3: Write minimal implementation**

```python
# manylatents/lightning/hf_trainer.py
"""HuggingFace model wrapper for PyTorch Lightning."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import CausalLMOutput


@dataclass
class HFTrainerConfig:
    """Configuration for HFTrainerModule.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps for scheduler
        adam_epsilon: Epsilon for AdamW
        torch_dtype: Optional dtype for model (e.g., torch.bfloat16)
        trust_remote_code: Whether to trust remote code for model loading
    """
    model_name_or_path: str
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None


class HFTrainerModule(LightningModule):
    """Lightning module wrapping HuggingFace causal LM.

    Features:
    - Lazy model initialization (for FSDP compatibility)
    - Exposes .network for activation extraction
    - Standard Lightning training interface
    """

    def __init__(self, config: HFTrainerConfig):
        super().__init__()
        self.config = config
        self.network: Optional[AutoModelForCausalLM] = None
        self.tokenizer = None

        self.save_hyperparameters({"config": config.__dict__})

    def configure_model(self) -> None:
        """Lazy model initialization for FSDP compatibility."""
        if self.network is not None:
            return

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name_or_path,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.config.torch_dtype
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.network = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )

    def forward(self, **inputs) -> CausalLMOutput:
        """Forward pass through the model."""
        return self.network(**inputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard training step."""
        outputs: CausalLMOutput = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Standard validation step."""
        outputs: CausalLMOutput = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW with optional warmup."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        if self.config.warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/tests/test_hf_trainer.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/hf_trainer.py manylatents/lightning/tests/test_hf_trainer.py
git commit -m "feat(lightning): add HFTrainerModule for HuggingFace models"
```

---

### Task 8: HFTrainerModule with tiny model integration test

**Files:**
- Modify: `manylatents/lightning/tests/test_hf_trainer.py`

**Step 1: Write integration test with tiny model**

```python
# Add to manylatents/lightning/tests/test_hf_trainer.py
from lightning import Trainer


@pytest.mark.slow
def test_hf_trainer_module_forward_pass():
    """Integration test with actual tiny model."""
    config = HFTrainerConfig(
        model_name_or_path="sshleifer/tiny-gpt2",  # ~2MB model
        trust_remote_code=True,
    )
    module = HFTrainerModule(config)
    module.configure_model()

    # Create dummy batch
    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world", "Test input"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    # Forward pass
    module.eval()
    with torch.no_grad():
        outputs = module(**batch)

    assert outputs.loss is not None
    assert outputs.logits is not None


@pytest.mark.slow
def test_hf_trainer_module_training_step():
    """Test training step computes loss."""
    config = HFTrainerConfig(model_name_or_path="sshleifer/tiny-gpt2")
    module = HFTrainerModule(config)
    module.configure_model()

    tokenizer = module.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hello world"],
        return_tensors="pt",
        padding=True,
        max_length=32,
    )
    batch["labels"] = batch["input_ids"].clone()

    loss = module.training_step(batch, 0)

    assert loss is not None
    assert loss.requires_grad
```

**Step 2: Run tests**

Run: `pytest manylatents/lightning/tests/test_hf_trainer.py -v -m slow`
Expected: PASS (2 slow tests, may download tiny model first time)

**Step 3: Commit**

```bash
git add manylatents/lightning/tests/test_hf_trainer.py
git commit -m "test(lightning): add HFTrainerModule integration tests"
```

---

## Phase 4: RepresentationAuditCallback

### Task 9: AuditTrigger config

**Files:**
- Create: `manylatents/lightning/callbacks/__init__.py`
- Create: `manylatents/lightning/callbacks/audit.py`
- Test: `manylatents/lightning/callbacks/tests/test_audit.py`

**Step 1: Write the failing test**

```python
# manylatents/lightning/callbacks/tests/test_audit.py
import pytest
from manylatents.lightning.callbacks.audit import AuditTrigger


def test_audit_trigger_step_based():
    trigger = AuditTrigger(every_n_steps=100)

    assert trigger.should_fire(step=0, epoch=0) is True   # First step
    assert trigger.should_fire(step=50, epoch=0) is False
    assert trigger.should_fire(step=100, epoch=0) is True
    assert trigger.should_fire(step=200, epoch=0) is True


def test_audit_trigger_epoch_based():
    trigger = AuditTrigger(every_n_epochs=2)

    assert trigger.should_fire(step=0, epoch=0, epoch_end=True) is True
    assert trigger.should_fire(step=0, epoch=1, epoch_end=True) is False
    assert trigger.should_fire(step=0, epoch=2, epoch_end=True) is True


def test_audit_trigger_combined():
    trigger = AuditTrigger(every_n_steps=100, every_n_epochs=1)

    # Steps trigger
    assert trigger.should_fire(step=100, epoch=0) is True
    # Epoch also triggers
    assert trigger.should_fire(step=50, epoch=1, epoch_end=True) is True


def test_audit_trigger_disabled():
    trigger = AuditTrigger()  # No triggers set

    assert trigger.should_fire(step=100, epoch=5) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py::test_audit_trigger_step_based -v`
Expected: FAIL with "No module named 'manylatents.lightning.callbacks'"

**Step 3: Create package and implementation**

```python
# manylatents/lightning/callbacks/__init__.py
"""Lightning callbacks for representation auditing."""
from manylatents.lightning.callbacks.audit import (
    AuditTrigger,
    RepresentationAuditCallback,
)

__all__ = ["AuditTrigger", "RepresentationAuditCallback"]
```

```python
# manylatents/lightning/callbacks/tests/__init__.py
# Empty init
```

```python
# manylatents/lightning/callbacks/audit.py
"""Representation auditing callback for Lightning."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader

from manylatents.lightning.hooks import ActivationExtractor, LayerSpec
from manylatents.gauge.diffusion import DiffusionGauge


@dataclass
class AuditTrigger:
    """Configuration for when to trigger representation audits.

    Multiple triggers can be combined (OR logic).

    Attributes:
        every_n_steps: Trigger every N training steps
        every_n_epochs: Trigger at the end of every N epochs
        on_checkpoint: Trigger when checkpoint is saved
        on_validation_end: Trigger after validation
    """
    every_n_steps: Optional[int] = None
    every_n_epochs: Optional[int] = None
    on_checkpoint: bool = False
    on_validation_end: bool = False

    def should_fire(
        self,
        step: int,
        epoch: int,
        epoch_end: bool = False,
        checkpoint: bool = False,
        validation_end: bool = False,
    ) -> bool:
        """Check if audit should trigger based on current state."""
        if self.every_n_steps is not None:
            if step % self.every_n_steps == 0:
                return True

        if self.every_n_epochs is not None and epoch_end:
            if epoch % self.every_n_epochs == 0:
                return True

        if self.on_checkpoint and checkpoint:
            return True

        if self.on_validation_end and validation_end:
            return True

        return False
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/callbacks/
git commit -m "feat(lightning): add AuditTrigger configuration"
```

---

### Task 10: RepresentationAuditCallback core

**Files:**
- Modify: `manylatents/lightning/callbacks/audit.py`
- Modify: `manylatents/lightning/callbacks/tests/test_audit.py`

**Step 1: Write the failing test**

```python
# Add to manylatents/lightning/callbacks/tests/test_audit.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, Trainer
from manylatents.lightning.callbacks.audit import (
    AuditTrigger,
    RepresentationAuditCallback,
)
from manylatents.lightning.hooks import LayerSpec


class TinyModel(LightningModule):
    """Minimal model for testing."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_probe_loader(n_samples=20, input_dim=10):
    x = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 5)
    return DataLoader(TensorDataset(x, y), batch_size=10)


def test_representation_audit_callback_captures():
    """Callback should capture activations and compute diffusion ops."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationAuditCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="network.0", reduce="none")],
        trigger=AuditTrigger(every_n_steps=1),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    # Should have captured at least one trajectory point
    trajectory = callback.get_trajectory()
    assert len(trajectory) > 0

    # Each point should have step and diffusion operator
    step, diff_op = trajectory[0]
    assert isinstance(step, int)
    assert isinstance(diff_op, np.ndarray)
    assert diff_op.shape == (20, 20)  # probe_loader has 20 samples
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py::test_representation_audit_callback_captures -v`
Expected: FAIL (RepresentationAuditCallback not fully implemented)

**Step 3: Implement RepresentationAuditCallback**

```python
# Add to manylatents/lightning/callbacks/audit.py (after AuditTrigger class)

@dataclass
class TrajectoryPoint:
    """A single point in the representation trajectory."""
    step: int
    epoch: int
    diffusion_operators: Dict[str, np.ndarray]  # layer_path -> operator
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepresentationAuditCallback(Callback):
    """Callback that extracts activations and computes diffusion operators.

    At configurable triggers during training, this callback:
    1. Runs the probe set through the model
    2. Extracts activations from specified layers
    3. Computes diffusion operators from activations
    4. Stores (step, operator) pairs in a trajectory

    Usage:
        callback = RepresentationAuditCallback(
            probe_loader=probe_loader,
            layer_specs=[LayerSpec("model.layers[-1]")],
            trigger=AuditTrigger(every_n_steps=100),
        )
        trainer = Trainer(callbacks=[callback])
        trainer.fit(model)

        trajectory = callback.get_trajectory()
    """

    def __init__(
        self,
        probe_loader: DataLoader,
        layer_specs: List[LayerSpec],
        trigger: AuditTrigger,
        gauge: Optional[DiffusionGauge] = None,
    ):
        super().__init__()
        self.probe_loader = probe_loader
        self.layer_specs = layer_specs
        self.trigger = trigger
        self.gauge = gauge or DiffusionGauge()

        self.extractor = ActivationExtractor(layer_specs)
        self._trajectory: List[TrajectoryPoint] = []

    def _extract_and_gauge(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> Dict[str, np.ndarray]:
        """Extract activations and compute diffusion operators."""
        # Get the underlying network
        network = getattr(pl_module, 'network', pl_module)

        # Run probe set through model
        network.eval()
        with torch.no_grad():
            with self.extractor.capture(network):
                for batch in self.probe_loader:
                    # Handle different batch formats
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    elif isinstance(batch, dict):
                        inputs = batch
                    else:
                        inputs = batch

                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(pl_module.device)
                        network(inputs)
                    else:
                        # Dict input (for HF models)
                        inputs = {k: v.to(pl_module.device) for k, v in inputs.items()}
                        network(**inputs)

        network.train()

        # Get activations and compute diffusion operators
        activations = self.extractor.get_activations()
        diffusion_ops = {}
        for path, acts in activations.items():
            diffusion_ops[path] = self.gauge(acts)

        return diffusion_ops

    def _maybe_audit(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        epoch_end: bool = False,
        checkpoint: bool = False,
        validation_end: bool = False,
    ):
        """Check trigger and perform audit if needed."""
        if not self.trigger.should_fire(
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            epoch_end=epoch_end,
            checkpoint=checkpoint,
            validation_end=validation_end,
        ):
            return

        diff_ops = self._extract_and_gauge(trainer, pl_module)

        point = TrajectoryPoint(
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            diffusion_operators=diff_ops,
        )
        self._trajectory.append(point)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Check step-based triggers."""
        self._maybe_audit(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Check epoch-based triggers."""
        self._maybe_audit(trainer, pl_module, epoch_end=True)

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ):
        """Check validation-end trigger."""
        self._maybe_audit(trainer, pl_module, validation_end=True)

    def get_trajectory(self) -> List[tuple]:
        """Get trajectory as list of (step, diffusion_operator) tuples.

        For single-layer extraction, returns the operator directly.
        For multi-layer, returns dict of operators.
        """
        result = []
        for point in self._trajectory:
            if len(point.diffusion_operators) == 1:
                # Single layer - return operator directly
                op = list(point.diffusion_operators.values())[0]
                result.append((point.step, op))
            else:
                result.append((point.step, point.diffusion_operators))
        return result

    def get_full_trajectory(self) -> List[TrajectoryPoint]:
        """Get full trajectory with metadata."""
        return self._trajectory.copy()

    def clear(self):
        """Clear stored trajectory."""
        self._trajectory.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py::test_representation_audit_callback_captures -v`
Expected: PASS

**Step 5: Commit**

```bash
git add manylatents/lightning/callbacks/audit.py manylatents/lightning/callbacks/tests/test_audit.py
git commit -m "feat(lightning): add RepresentationAuditCallback for training-time auditing"
```

---

### Task 11: RepresentationAuditCallback multi-layer test

**Files:**
- Modify: `manylatents/lightning/callbacks/tests/test_audit.py`

**Step 1: Write multi-layer test**

```python
# Add to manylatents/lightning/callbacks/tests/test_audit.py

def test_representation_audit_callback_multi_layer():
    """Should capture multiple layers."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationAuditCallback(
        probe_loader=probe_loader,
        layer_specs=[
            LayerSpec(path="network.0", reduce="none"),  # First linear
            LayerSpec(path="network.2", reduce="none"),  # Second linear
        ],
        trigger=AuditTrigger(every_n_steps=2),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=1,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    trajectory = callback.get_trajectory()
    assert len(trajectory) > 0

    # Multi-layer returns dict
    step, ops = trajectory[0]
    assert isinstance(ops, dict)
    assert "network.0" in ops
    assert "network.2" in ops


def test_representation_audit_callback_epoch_trigger():
    """Should trigger at epoch end."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationAuditCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="network.0", reduce="none")],
        trigger=AuditTrigger(every_n_epochs=1),
    )

    train_loader = make_probe_loader(n_samples=40)

    trainer = Trainer(
        max_epochs=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_loader)

    trajectory = callback.get_full_trajectory()
    # Should have captured at epoch 0 and epoch 1
    epochs = [p.epoch for p in trajectory]
    assert 0 in epochs
    assert 1 in epochs
```

**Step 2: Run tests**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py -v`
Expected: PASS (6 tests)

**Step 3: Commit**

```bash
git add manylatents/lightning/callbacks/tests/test_audit.py
git commit -m "test(lightning): add multi-layer and epoch trigger tests for audit callback"
```

---

## Phase 5: Diffusion Operator Merging

### Task 12: DiffusionMerging strategies

**Files:**
- Modify: `manylatents/algorithms/latent/merging.py`
- Test: `manylatents/algorithms/latent/tests/test_diffusion_merging.py`

**Step 1: Write the failing test**

```python
# manylatents/algorithms/latent/tests/test_diffusion_merging.py
import pytest
import numpy as np
from manylatents.algorithms.latent.merging import DiffusionMerging


def make_random_diffusion_op(n: int, seed: int) -> np.ndarray:
    """Create a random row-stochastic matrix."""
    rng = np.random.default_rng(seed)
    K = rng.random((n, n))
    K = (K + K.T) / 2  # Symmetric kernel
    np.fill_diagonal(K, 0)
    row_sums = K.sum(axis=1, keepdims=True)
    return K / row_sums


def test_diffusion_merging_weighted_interpolation():
    """Weighted interpolation of operators."""
    ops = {
        "model_a": make_random_diffusion_op(50, seed=1),
        "model_b": make_random_diffusion_op(50, seed=2),
    }

    merger = DiffusionMerging(strategy="weighted_interpolation")
    merged = merger.merge(ops)

    assert merged.shape == (50, 50)
    # Should be row-stochastic
    row_sums = merged.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_merging_frobenius_mean():
    """Frobenius mean of operators."""
    ops = {
        "model_a": make_random_diffusion_op(50, seed=1),
        "model_b": make_random_diffusion_op(50, seed=2),
        "model_c": make_random_diffusion_op(50, seed=3),
    }

    merger = DiffusionMerging(strategy="frobenius_mean")
    merged = merger.merge(ops)

    assert merged.shape == (50, 50)
    # Frobenius mean is arithmetic mean
    expected = (ops["model_a"] + ops["model_b"] + ops["model_c"]) / 3
    np.testing.assert_allclose(merged, expected)


def test_diffusion_merging_with_weights():
    """Weighted interpolation with custom weights."""
    ops = {
        "model_a": make_random_diffusion_op(30, seed=1),
        "model_b": make_random_diffusion_op(30, seed=2),
    }

    merger = DiffusionMerging(
        strategy="weighted_interpolation",
        weights={"model_a": 0.8, "model_b": 0.2},
    )
    merged = merger.merge(ops)

    # Merged should be closer to model_a
    dist_to_a = np.linalg.norm(merged - ops["model_a"], "fro")
    dist_to_b = np.linalg.norm(merged - ops["model_b"], "fro")
    assert dist_to_a < dist_to_b
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/algorithms/latent/tests/test_diffusion_merging.py -v`
Expected: FAIL with "cannot import name 'DiffusionMerging'"

**Step 3: Implement DiffusionMerging**

```python
# Add to manylatents/algorithms/latent/merging.py (at end of file)

class DiffusionMerging:
    """Merge multiple diffusion operators into a single target operator.

    Strategies:
    - weighted_interpolation: P* = Σ w_i P_i, then normalize to row-stochastic
    - frobenius_mean: P* = (1/N) Σ P_i (arithmetic mean, closed-form Frobenius)
    - ot_barycenter: Wasserstein barycenter (requires POT library)

    Attributes:
        strategy: Merging strategy
        weights: Optional per-operator weights (normalized internally)
        normalize_output: Whether to ensure output is row-stochastic
    """

    STRATEGIES = ("weighted_interpolation", "frobenius_mean", "ot_barycenter")

    def __init__(
        self,
        strategy: str = "weighted_interpolation",
        weights: Optional[Dict[str, float]] = None,
        normalize_output: bool = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.strategy = strategy
        self.weights = weights or {}
        self.normalize_output = normalize_output

    def merge(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Merge multiple diffusion operators.

        Args:
            operators: Dict mapping operator name to (N, N) array

        Returns:
            Merged operator of shape (N, N)
        """
        if len(operators) == 0:
            raise ValueError("operators dict is empty")

        # Validate all same shape
        shapes = {k: v.shape for k, v in operators.items()}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(f"All operators must have same shape, got {shapes}")

        if self.strategy == "weighted_interpolation":
            return self._weighted_interpolation(operators)
        elif self.strategy == "frobenius_mean":
            return self._frobenius_mean(operators)
        elif self.strategy == "ot_barycenter":
            return self._ot_barycenter(operators)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_weights(self, operators: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get normalized weights for operators."""
        weights = {k: self.weights.get(k, 1.0) for k in operators}
        total = sum(weights.values())
        return {k: w / total for k, w in weights.items()}

    def _weighted_interpolation(
        self, operators: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Weighted sum of operators, normalized to row-stochastic."""
        weights = self._get_weights(operators)

        merged = sum(w * operators[k] for k, w in weights.items())

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged

    def _frobenius_mean(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Arithmetic mean (closed-form Frobenius barycenter)."""
        ops = list(operators.values())
        merged = np.mean(ops, axis=0)

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged

    def _ot_barycenter(self, operators: Dict[str, np.ndarray]) -> np.ndarray:
        """Wasserstein barycenter using POT library."""
        try:
            import ot
        except ImportError:
            raise ImportError(
                "ot_barycenter strategy requires POT library. "
                "Install with: pip install POT"
            )

        weights = self._get_weights(operators)
        ops = list(operators.values())
        weight_array = np.array([weights[k] for k in operators])

        # Stack operators for POT
        # POT expects distributions as columns
        n = ops[0].shape[0]

        # Use Sinkhorn barycenter on rows
        # Each row of the diffusion operator is a distribution
        merged_rows = []
        for i in range(n):
            # Get i-th row from each operator
            distributions = np.array([op[i, :] for op in ops]).T  # (n, num_ops)

            # Compute barycenter of these distributions
            M = ot.dist(np.arange(n).reshape(-1, 1).astype(float))  # Cost matrix
            M = M / M.max()

            barycenter = ot.bregman.barycenter(
                distributions,
                M,
                reg=0.01,  # Regularization
                weights=weight_array,
            )
            merged_rows.append(barycenter)

        merged = np.array(merged_rows)

        if self.normalize_output:
            row_sums = merged.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            merged = merged / row_sums

        return merged
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/algorithms/latent/tests/test_diffusion_merging.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add manylatents/algorithms/latent/merging.py manylatents/algorithms/latent/tests/test_diffusion_merging.py
git commit -m "feat(merging): add DiffusionMerging for operator-level fusion"
```

---

### Task 13: DiffusionMerging OT barycenter test

**Files:**
- Modify: `manylatents/algorithms/latent/tests/test_diffusion_merging.py`

**Step 1: Write OT test (skipped if POT not installed)**

```python
# Add to manylatents/algorithms/latent/tests/test_diffusion_merging.py

pytest_plugins = ['pytest_dependency']


@pytest.mark.skipif(
    not pytest.importorskip("ot", reason="POT not installed"),
    reason="POT not installed"
)
def test_diffusion_merging_ot_barycenter():
    """OT barycenter of operators."""
    ops = {
        "model_a": make_random_diffusion_op(20, seed=1),
        "model_b": make_random_diffusion_op(20, seed=2),
    }

    merger = DiffusionMerging(strategy="ot_barycenter")
    merged = merger.merge(ops)

    assert merged.shape == (20, 20)
    # Should be row-stochastic
    row_sums = merged.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


def test_diffusion_merging_ot_import_error():
    """Should raise helpful error if POT not installed."""
    # This test verifies the error message, not that POT is missing
    # We mock the import to test the error path
    import sys

    # Temporarily hide ot module
    ot_module = sys.modules.get("ot")
    sys.modules["ot"] = None

    try:
        ops = {"a": make_random_diffusion_op(10, seed=1)}
        merger = DiffusionMerging(strategy="ot_barycenter")

        with pytest.raises(ImportError, match="POT library"):
            merger.merge(ops)
    finally:
        # Restore
        if ot_module is not None:
            sys.modules["ot"] = ot_module
        else:
            sys.modules.pop("ot", None)
```

**Step 2: Run tests**

Run: `pytest manylatents/algorithms/latent/tests/test_diffusion_merging.py -v`
Expected: PASS (4-5 tests depending on POT availability)

**Step 3: Commit**

```bash
git add manylatents/algorithms/latent/tests/test_diffusion_merging.py
git commit -m "test(merging): add OT barycenter tests for DiffusionMerging"
```

---

## Phase 6: Trajectory Visualization & PHATE Integration

### Task 14: TrajectoryVisualizer with PHATE

**Files:**
- Create: `manylatents/gauge/trajectory.py`
- Test: `manylatents/gauge/tests/test_trajectory.py`

**Step 1: Write the failing test**

```python
# manylatents/gauge/tests/test_trajectory.py
import pytest
import numpy as np
from manylatents.gauge.trajectory import TrajectoryVisualizer


def make_trajectory(n_steps: int, n_samples: int, seed: int = 42):
    """Create a fake trajectory of diffusion operators."""
    rng = np.random.default_rng(seed)
    trajectory = []
    for step in range(n_steps):
        # Gradually changing operator
        K = rng.random((n_samples, n_samples)) + step * 0.1
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        row_sums = K.sum(axis=1, keepdims=True)
        P = K / row_sums
        trajectory.append((step * 100, P))
    return trajectory


def test_trajectory_visualizer_embed():
    """Should embed trajectory points."""
    trajectory = make_trajectory(n_steps=10, n_samples=30)

    viz = TrajectoryVisualizer(n_components=2)
    embedding = viz.fit_transform(trajectory)

    assert embedding.shape == (10, 2)  # 10 steps, 2 dims


def test_trajectory_visualizer_distances():
    """Should compute pairwise distances between operators."""
    trajectory = make_trajectory(n_steps=5, n_samples=20)

    viz = TrajectoryVisualizer()
    distances = viz.compute_distances(trajectory)

    assert distances.shape == (5, 5)
    # Diagonal should be zero
    np.testing.assert_allclose(np.diag(distances), 0, atol=1e-10)
    # Should be symmetric
    np.testing.assert_allclose(distances, distances.T)
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/gauge/tests/test_trajectory.py -v`
Expected: FAIL

**Step 3: Implement TrajectoryVisualizer**

```python
# manylatents/gauge/trajectory.py
"""Visualize representation trajectories using PHATE."""
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import squareform, pdist

from manylatents.algorithms.latent.phate import PHATEModule


@dataclass
class TrajectoryVisualizer:
    """Embed diffusion operator trajectories for visualization.

    Takes a sequence of (step, operator) pairs and embeds them
    in low-dimensional space using PHATE on pairwise distances.

    Distance metrics:
    - frobenius: ||P_i - P_j||_F
    - spectral: Distance between eigenvalue spectra

    Attributes:
        n_components: Output embedding dimension
        distance_metric: How to measure operator distance
        phate_knn: k for PHATE k-NN graph
        phate_t: Diffusion time for PHATE
    """
    n_components: int = 2
    distance_metric: Literal["frobenius", "spectral"] = "frobenius"
    phate_knn: int = 5
    phate_t: int = 10

    def compute_distances(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """Compute pairwise distances between operators in trajectory."""
        operators = [op for _, op in trajectory]
        n = len(operators)

        if self.distance_metric == "frobenius":
            # Flatten and use pdist
            flat = [op.flatten() for op in operators]
            return squareform(pdist(flat, metric="euclidean"))

        elif self.distance_metric == "spectral":
            # Compare eigenvalue spectra
            spectra = []
            for op in operators:
                # Get top eigenvalues (sorted descending)
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(eigvals)[::-1]
                spectra.append(eigvals)
            return squareform(pdist(spectra, metric="euclidean"))

        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def fit_transform(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """Embed trajectory in low-dimensional space.

        Args:
            trajectory: List of (step, operator) tuples

        Returns:
            Array of shape (n_steps, n_components)
        """
        distances = self.compute_distances(trajectory)

        # Convert distance matrix to similarity for PHATE
        # Use Gaussian kernel
        sigma = np.median(distances[distances > 0])
        similarities = np.exp(-distances**2 / (2 * sigma**2))

        # Use PHATE on the similarity matrix
        phate = PHATEModule(
            n_components=self.n_components,
            knn=min(self.phate_knn, len(trajectory) - 1),
            t=self.phate_t,
        )

        # PHATE expects features, we'll use the distance matrix rows as features
        phate.fit(similarities)
        embedding = phate.transform(similarities)

        return embedding.numpy() if hasattr(embedding, 'numpy') else np.array(embedding)

    def compute_spread(
        self,
        trajectory: List[Tuple[int, np.ndarray]],
    ) -> float:
        """Compute spread metric (average pairwise distance)."""
        distances = self.compute_distances(trajectory)
        # Upper triangle only (excluding diagonal)
        upper_tri = distances[np.triu_indices(len(trajectory), k=1)]
        return float(np.mean(upper_tri))
```

**Step 4: Run test to verify it passes**

Run: `pytest manylatents/gauge/tests/test_trajectory.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add manylatents/gauge/trajectory.py manylatents/gauge/tests/test_trajectory.py
git commit -m "feat(gauge): add TrajectoryVisualizer for operator trajectory embedding"
```

---

### Task 15: Spread convergence metric

**Files:**
- Modify: `manylatents/gauge/trajectory.py`
- Modify: `manylatents/gauge/tests/test_trajectory.py`

**Step 1: Add convergence tests**

```python
# Add to manylatents/gauge/tests/test_trajectory.py

def make_converging_trajectories(
    n_models: int,
    n_steps: int,
    n_samples: int,
    seed: int = 42,
) -> List[List[Tuple[int, np.ndarray]]]:
    """Create trajectories that converge over time."""
    rng = np.random.default_rng(seed)

    # Target operator that all models converge to
    K_target = rng.random((n_samples, n_samples))
    K_target = (K_target + K_target.T) / 2
    np.fill_diagonal(K_target, 0)
    P_target = K_target / K_target.sum(axis=1, keepdims=True)

    trajectories = []
    for model_idx in range(n_models):
        trajectory = []
        # Start with random operator
        K_init = rng.random((n_samples, n_samples))
        K_init = (K_init + K_init.T) / 2
        np.fill_diagonal(K_init, 0)
        P_init = K_init / K_init.sum(axis=1, keepdims=True)

        for step in range(n_steps):
            # Interpolate toward target
            alpha = step / (n_steps - 1) if n_steps > 1 else 1.0
            P = (1 - alpha) * P_init + alpha * P_target
            trajectory.append((step * 100, P))

        trajectories.append(trajectory)

    return trajectories


def test_trajectory_spread_convergence():
    """Spread should decrease for converging trajectories."""
    from manylatents.gauge.trajectory import compute_multi_model_spread

    trajectories = make_converging_trajectories(
        n_models=3, n_steps=10, n_samples=20
    )

    spreads = compute_multi_model_spread(trajectories)

    # Spread at later steps should be lower
    assert spreads[-1] < spreads[0]
```

**Step 2: Implement multi-model spread**

```python
# Add to manylatents/gauge/trajectory.py

def compute_multi_model_spread(
    trajectories: List[List[Tuple[int, np.ndarray]]],
    distance_metric: str = "frobenius",
) -> np.ndarray:
    """Compute spread across models at each timestep.

    For each timestep t, computes the average pairwise distance
    between operators P_{i,t} across models i.

    Args:
        trajectories: List of trajectories, one per model.
            Each trajectory is List[(step, operator)]
        distance_metric: "frobenius" or "spectral"

    Returns:
        Array of shape (n_steps,) with spread at each step
    """
    n_models = len(trajectories)
    n_steps = len(trajectories[0])

    spreads = []
    for t in range(n_steps):
        # Get operators at time t from all models
        ops_at_t = [traj[t][1] for traj in trajectories]

        # Compute pairwise distances
        if distance_metric == "frobenius":
            dists = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    dist = np.linalg.norm(ops_at_t[i] - ops_at_t[j], "fro")
                    dists.append(dist)
            spread = np.mean(dists) if dists else 0.0
        else:
            # spectral
            spectra = []
            for op in ops_at_t:
                eigvals = np.sort(np.linalg.eigvalsh(op))[::-1]
                spectra.append(eigvals)
            dists = squareform(pdist(spectra))
            spread = np.mean(dists[np.triu_indices(n_models, k=1)])

        spreads.append(spread)

    return np.array(spreads)
```

**Step 3: Run tests**

Run: `pytest manylatents/gauge/tests/test_trajectory.py -v`
Expected: PASS (3 tests)

**Step 4: Commit**

```bash
git add manylatents/gauge/trajectory.py manylatents/gauge/tests/test_trajectory.py
git commit -m "feat(gauge): add compute_multi_model_spread for convergence analysis"
```

---

## Phase 7: WandB Integration

### Task 16: WandB logging for audit callback

**Files:**
- Create: `manylatents/lightning/callbacks/wandb_audit.py`
- Test: `manylatents/lightning/callbacks/tests/test_wandb_audit.py`

**Step 1: Write the failing test**

```python
# manylatents/lightning/callbacks/tests/test_wandb_audit.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from manylatents.lightning.callbacks.wandb_audit import WandbAuditLogger


def test_wandb_audit_logger_logs_spread():
    """Should log spread metric to wandb."""
    trajectory = [
        (0, np.eye(10)),
        (100, np.eye(10) * 0.9),
    ]

    with patch("wandb.log") as mock_log:
        logger = WandbAuditLogger(log_spread=True)
        logger.log_trajectory(trajectory)

        # Should have logged spread
        mock_log.assert_called()
        call_args = mock_log.call_args_list
        logged_keys = set()
        for call in call_args:
            logged_keys.update(call[0][0].keys())

        assert "audit/spread" in logged_keys or any("spread" in k for k in logged_keys)


def test_wandb_audit_logger_logs_operator_stats():
    """Should log operator statistics."""
    trajectory = [
        (0, np.eye(10)),
    ]

    with patch("wandb.log") as mock_log:
        logger = WandbAuditLogger(log_operator_stats=True)
        logger.log_trajectory(trajectory)

        mock_log.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest manylatents/lightning/callbacks/tests/test_wandb_audit.py -v`
Expected: FAIL

**Step 3: Implement WandbAuditLogger**

```python
# manylatents/lightning/callbacks/wandb_audit.py
"""WandB logging for representation auditing."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class WandbAuditLogger:
    """Log representation audit results to WandB.

    Logs:
    - Trajectory spread over time
    - Operator statistics (spectral properties)
    - Optional: Operator heatmaps as images

    Attributes:
        log_spread: Log spread metrics
        log_operator_stats: Log operator statistics
        log_images: Log operator heatmaps as images
        prefix: Prefix for all logged keys
    """
    log_spread: bool = True
    log_operator_stats: bool = True
    log_images: bool = False
    prefix: str = "audit"

    def log_trajectory(
        self,
        trajectory: List[Tuple[int, np.ndarray]],
        step: Optional[int] = None,
    ):
        """Log trajectory statistics to wandb.

        Args:
            trajectory: List of (step, operator) tuples
            step: Global step for logging (uses last trajectory step if None)
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        if not trajectory:
            return

        log_step = step if step is not None else trajectory[-1][0]
        metrics = {}

        if self.log_spread and len(trajectory) > 1:
            spread = self._compute_spread(trajectory)
            metrics[f"{self.prefix}/spread"] = spread

        if self.log_operator_stats:
            # Stats for latest operator
            _, latest_op = trajectory[-1]
            stats = self._compute_operator_stats(latest_op)
            for k, v in stats.items():
                metrics[f"{self.prefix}/{k}"] = v

        if self.log_images and len(trajectory) > 0:
            _, latest_op = trajectory[-1]
            metrics[f"{self.prefix}/operator"] = wandb.Image(
                latest_op,
                caption=f"Diffusion operator at step {log_step}",
            )

        wandb.log(metrics, step=log_step)

    def log_multi_model_spread(
        self,
        trajectories: List[List[Tuple[int, np.ndarray]]],
        model_names: Optional[List[str]] = None,
        step: Optional[int] = None,
    ):
        """Log spread across multiple models at current step.

        Args:
            trajectories: List of trajectories, one per model
            model_names: Optional names for models
            step: Global step for logging
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        from manylatents.gauge.trajectory import compute_multi_model_spread

        spreads = compute_multi_model_spread(trajectories)

        metrics = {}
        for i, spread in enumerate(spreads):
            metrics[f"{self.prefix}/multi_model_spread_step_{i}"] = spread

        # Also log latest spread
        if len(spreads) > 0:
            metrics[f"{self.prefix}/multi_model_spread_latest"] = spreads[-1]

        wandb.log(metrics, step=step)

    def _compute_spread(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> float:
        """Compute spread of operators in trajectory."""
        if len(trajectory) < 2:
            return 0.0

        ops = [op for _, op in trajectory]
        dists = []
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                dists.append(np.linalg.norm(ops[i] - ops[j], "fro"))
        return float(np.mean(dists))

    def _compute_operator_stats(self, op: np.ndarray) -> Dict[str, float]:
        """Compute statistics of a diffusion operator."""
        stats = {}

        # Spectral properties
        eigvals = np.linalg.eigvalsh(op)
        stats["spectral_gap"] = float(1.0 - np.sort(np.abs(eigvals))[-2])
        stats["spectral_radius"] = float(np.max(np.abs(eigvals)))

        # Entropy of stationary distribution (if row-stochastic)
        row_sums = op.sum(axis=1)
        if np.allclose(row_sums, 1.0, rtol=1e-3):
            # Approximate stationary distribution
            pi = np.ones(len(op)) / len(op)
            for _ in range(100):
                pi = pi @ op
            pi = pi / pi.sum()
            entropy = -np.sum(pi * np.log(pi + 1e-10))
            stats["stationary_entropy"] = float(entropy)

        return stats
```

**Step 4: Run tests**

Run: `pytest manylatents/lightning/callbacks/tests/test_wandb_audit.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add manylatents/lightning/callbacks/wandb_audit.py manylatents/lightning/callbacks/tests/test_wandb_audit.py
git commit -m "feat(lightning): add WandbAuditLogger for trajectory logging"
```

---

### Task 17: Integrate WandbAuditLogger with RepresentationAuditCallback

**Files:**
- Modify: `manylatents/lightning/callbacks/audit.py`
- Modify: `manylatents/lightning/callbacks/tests/test_audit.py`

**Step 1: Add wandb integration test**

```python
# Add to manylatents/lightning/callbacks/tests/test_audit.py
from unittest.mock import patch, MagicMock


def test_representation_audit_callback_with_wandb():
    """Should log to wandb when enabled."""
    model = TinyModel()
    probe_loader = make_probe_loader()

    callback = RepresentationAuditCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="network.0", reduce="none")],
        trigger=AuditTrigger(every_n_steps=2),
        log_to_wandb=True,
    )

    train_loader = make_probe_loader(n_samples=40)

    with patch("wandb.run", MagicMock()), \
         patch("wandb.log") as mock_log:

        trainer = Trainer(
            max_epochs=1,
            callbacks=[callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, train_loader)

        # Should have logged at least once
        assert mock_log.called
```

**Step 2: Modify RepresentationAuditCallback**

```python
# Modify RepresentationAuditCallback.__init__ in manylatents/lightning/callbacks/audit.py

def __init__(
    self,
    probe_loader: DataLoader,
    layer_specs: List[LayerSpec],
    trigger: AuditTrigger,
    gauge: Optional[DiffusionGauge] = None,
    log_to_wandb: bool = False,
    wandb_prefix: str = "audit",
):
    super().__init__()
    self.probe_loader = probe_loader
    self.layer_specs = layer_specs
    self.trigger = trigger
    self.gauge = gauge or DiffusionGauge()
    self.log_to_wandb = log_to_wandb

    self.extractor = ActivationExtractor(layer_specs)
    self._trajectory: List[TrajectoryPoint] = []

    if log_to_wandb:
        from manylatents.lightning.callbacks.wandb_audit import WandbAuditLogger
        self._wandb_logger = WandbAuditLogger(prefix=wandb_prefix)
    else:
        self._wandb_logger = None

# Modify _maybe_audit to log to wandb
def _maybe_audit(
    self,
    trainer: Trainer,
    pl_module: LightningModule,
    epoch_end: bool = False,
    checkpoint: bool = False,
    validation_end: bool = False,
):
    """Check trigger and perform audit if needed."""
    if not self.trigger.should_fire(
        step=trainer.global_step,
        epoch=trainer.current_epoch,
        epoch_end=epoch_end,
        checkpoint=checkpoint,
        validation_end=validation_end,
    ):
        return

    diff_ops = self._extract_and_gauge(trainer, pl_module)

    point = TrajectoryPoint(
        step=trainer.global_step,
        epoch=trainer.current_epoch,
        diffusion_operators=diff_ops,
    )
    self._trajectory.append(point)

    # Log to wandb
    if self._wandb_logger is not None:
        trajectory = self.get_trajectory()
        self._wandb_logger.log_trajectory(trajectory, step=trainer.global_step)
```

**Step 3: Run tests**

Run: `pytest manylatents/lightning/callbacks/tests/test_audit.py -v`
Expected: PASS (7 tests)

**Step 4: Commit**

```bash
git add manylatents/lightning/callbacks/audit.py manylatents/lightning/callbacks/tests/test_audit.py
git commit -m "feat(lightning): integrate WandbAuditLogger with RepresentationAuditCallback"
```

---

## Phase 8: Shop/SLURM Integration for Parallel Runs

### Task 18: Hydra config for representation audit experiments

**Files:**
- Create: `manylatents/configs/callbacks/audit/default.yaml`
- Create: `manylatents/configs/experiment/representation_audit.yaml`

**Step 1: Create callback config**

```yaml
# manylatents/configs/callbacks/audit/default.yaml
_target_: manylatents.lightning.callbacks.audit.RepresentationAuditCallback
_partial_: true

# Probe loader configured separately
probe_loader: null

layer_specs:
  - _target_: manylatents.lightning.hooks.LayerSpec
    path: "model.layers[-1]"
    extraction_point: "output"
    reduce: "mean"

trigger:
  _target_: manylatents.lightning.callbacks.audit.AuditTrigger
  every_n_steps: 100
  every_n_epochs: null
  on_checkpoint: false
  on_validation_end: true

gauge:
  _target_: manylatents.gauge.diffusion.DiffusionGauge
  knn: 15
  alpha: 1.0
  symmetric: false

log_to_wandb: true
wandb_prefix: "audit"
```

**Step 2: Create experiment config**

```yaml
# manylatents/configs/experiment/representation_audit.yaml
# @package _global_
defaults:
  - /data: wikitext  # Or your preferred text dataset
  - /algorithms/lightning: hf_trainer
  - /callbacks/audit: default
  - /logger: wandb

# Experiment settings
experiment_name: representation_audit

# Model configuration
algorithms:
  lightning:
    config:
      model_name_or_path: "gpt2"
      learning_rate: 2e-5
      warmup_steps: 100

# Audit configuration
callbacks:
  audit:
    trigger:
      every_n_steps: 500
    layer_specs:
      - path: "transformer.h[-1]"
        reduce: "mean"

# Training
trainer:
  max_epochs: 3

# Probe dataset (subset for auditing)
probe:
  dataset: ${data.dataset}
  n_samples: 1000
```

**Step 3: Commit**

```bash
git add manylatents/configs/callbacks/audit/ manylatents/configs/experiment/representation_audit.yaml
git commit -m "feat(configs): add Hydra configs for representation audit experiments"
```

---

### Task 19: Shop launcher configuration

**Files:**
- Create: `shop/configs/experiments/representation_audit_sweep.yaml`

**Step 1: Create sweep config for parallel runs**

```yaml
# shop/configs/experiments/representation_audit_sweep.yaml
# Multi-model convergence experiment via Shop launchers

defaults:
  - /hydra/launcher: remote_slurm  # or slurm for local cluster

# Sweep over model sizes and seeds
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      algorithms.lightning.config.model_name_or_path: "gpt2,gpt2-medium,gpt2-large"
      seed: "42,43,44,45,46"  # 5 seeds per model

# Cluster resources
launcher:
  partition: main
  gres: "gpu:1"
  cpus_per_task: 4
  mem_gb: 32
  time_limit: "04:00:00"

  # Array job for parallel execution
  array: "1-15"  # 3 models × 5 seeds

# Shop-specific settings
shop:
  project: lrw
  cluster: mila  # or narval, cedar

# WandB grouping for analysis
wandb:
  project: lrw-representation-audit
  group: "convergence_${now:%Y%m%d}"
  tags:
    - convergence
    - multi-seed
```

**Step 2: Create launcher script**

```python
# shop/scripts/launch_audit_sweep.py
"""Launch parallel representation audit experiments."""
import subprocess
import sys
from pathlib import Path


def launch_sweep(
    config: str = "representation_audit_sweep",
    cluster: str = "mila",
    dry_run: bool = False,
):
    """Launch a sweep of audit experiments.

    Args:
        config: Sweep config name (without .yaml)
        cluster: Target cluster (mila, narval, cedar, local)
        dry_run: If True, print command without executing
    """
    cmd = [
        "python", "-m", "manylatents.main",
        f"+experiment=representation_audit",
        f"hydra/launcher={cluster}_slurm",
        f"+shop/experiments={config}",
        "--multirun",
    ]

    if dry_run:
        print("Would run:", " ".join(cmd))
        return

    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="representation_audit_sweep")
    parser.add_argument("--cluster", default="mila")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    launch_sweep(args.config, args.cluster, args.dry_run)
```

**Step 3: Commit**

```bash
git add shop/configs/experiments/representation_audit_sweep.yaml shop/scripts/launch_audit_sweep.py
git commit -m "feat(shop): add launcher configs for parallel audit experiments"
```

---

### Task 20: Aggregation script for multi-run results

**Files:**
- Create: `manylatents/scripts/aggregate_trajectories.py`
- Test: `manylatents/scripts/tests/test_aggregate.py`

**Step 1: Write failing test**

```python
# manylatents/scripts/tests/test_aggregate.py
import pytest
import numpy as np
import tempfile
from pathlib import Path

from manylatents.scripts.aggregate_trajectories import (
    load_trajectories,
    aggregate_convergence_metrics,
)


def create_mock_trajectory_file(path: Path, n_steps: int = 10, n_samples: int = 20):
    """Create a mock trajectory .npz file."""
    trajectory = []
    for step in range(n_steps):
        K = np.random.rand(n_samples, n_samples)
        K = (K + K.T) / 2
        np.fill_diagonal(K, 0)
        P = K / K.sum(axis=1, keepdims=True)
        trajectory.append((step * 100, P))

    np.savez(path, trajectory=trajectory)


def test_load_trajectories():
    """Should load trajectories from multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock files
        for i in range(3):
            create_mock_trajectory_file(tmpdir / f"run_{i}.npz")

        trajectories = load_trajectories(tmpdir)

        assert len(trajectories) == 3


def test_aggregate_convergence_metrics():
    """Should compute convergence metrics across runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in range(3):
            create_mock_trajectory_file(tmpdir / f"run_{i}.npz", n_steps=5)

        trajectories = load_trajectories(tmpdir)
        metrics = aggregate_convergence_metrics(trajectories)

        assert "spread_over_time" in metrics
        assert "final_spread" in metrics
        assert len(metrics["spread_over_time"]) == 5
```

**Step 2: Implement aggregation script**

```python
# manylatents/scripts/aggregate_trajectories.py
"""Aggregate trajectory results from parallel runs."""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

from manylatents.gauge.trajectory import compute_multi_model_spread


def load_trajectories(
    results_dir: Path,
    pattern: str = "*.npz",
) -> List[List[Tuple[int, np.ndarray]]]:
    """Load trajectories from result files.

    Args:
        results_dir: Directory containing trajectory .npz files
        pattern: Glob pattern for files

    Returns:
        List of trajectories, one per run
    """
    trajectories = []
    for path in sorted(results_dir.glob(pattern)):
        data = np.load(path, allow_pickle=True)
        trajectory = data["trajectory"].tolist()
        trajectories.append(trajectory)
    return trajectories


def aggregate_convergence_metrics(
    trajectories: List[List[Tuple[int, np.ndarray]]],
) -> Dict[str, Any]:
    """Compute aggregated convergence metrics.

    Args:
        trajectories: List of trajectories from multiple runs

    Returns:
        Dict with:
        - spread_over_time: Array of spread at each step
        - final_spread: Spread at final step
        - convergence_rate: Rate of spread decrease
    """
    if not trajectories:
        return {}

    spreads = compute_multi_model_spread(trajectories)

    # Compute convergence rate (linear regression on log spread)
    steps = np.arange(len(spreads))
    log_spreads = np.log(spreads + 1e-10)

    if len(steps) > 1:
        slope, intercept = np.polyfit(steps, log_spreads, 1)
        convergence_rate = -slope  # Positive = converging
    else:
        convergence_rate = 0.0

    return {
        "spread_over_time": spreads,
        "final_spread": float(spreads[-1]) if len(spreads) > 0 else 0.0,
        "convergence_rate": float(convergence_rate),
        "n_runs": len(trajectories),
        "n_steps": len(spreads),
    }


def main():
    """CLI for aggregating trajectories."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--pattern", default="*.npz")
    args = parser.parse_args()

    trajectories = load_trajectories(args.results_dir, args.pattern)
    metrics = aggregate_convergence_metrics(trajectories)

    # Convert numpy arrays to lists for JSON
    metrics_json = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(metrics_json, indent=2))


if __name__ == "__main__":
    main()
```

**Step 3: Run tests**

Run: `pytest manylatents/scripts/tests/test_aggregate.py -v`
Expected: PASS (2 tests)

**Step 4: Commit**

```bash
git add manylatents/scripts/aggregate_trajectories.py manylatents/scripts/tests/test_aggregate.py
git commit -m "feat(scripts): add trajectory aggregation for multi-run analysis"
```

---

## Phase 9: End-to-End Integration Tests

### Task 21: E2E test for convergent trajectories experiment

**Files:**
- Create: `manylatents/tests/integration/test_convergent_trajectories.py`

**Step 1: Write E2E integration test**

```python
# manylatents/tests/integration/test_convergent_trajectories.py
"""End-to-end test for convergent trajectories experiment."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer

from manylatents.lightning.hf_trainer import HFTrainerModule, HFTrainerConfig
from manylatents.lightning.callbacks.audit import (
    RepresentationAuditCallback,
    AuditTrigger,
)
from manylatents.lightning.hooks import LayerSpec
from manylatents.gauge.trajectory import TrajectoryVisualizer, compute_multi_model_spread


class TinyLM(nn.Module):
    """Tiny language model for testing."""
    def __init__(self, vocab_size=100, hidden_dim=32, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=64,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)

        # Compute loss if labels provided
        loss = None
        if "labels" in kwargs:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                kwargs["labels"].view(-1),
            )

        # Return object with .loss attribute
        class Output:
            pass
        out = Output()
        out.loss = loss
        out.logits = logits
        return out


class TinyLMModule(nn.Module):
    """Wrapper that looks like HF model."""
    def __init__(self):
        super().__init__()
        self.model = TinyLM()

    def forward(self, **kwargs):
        return self.model(**kwargs)


@pytest.fixture
def probe_loader():
    """Create probe dataset."""
    input_ids = torch.randint(0, 100, (50, 16))
    labels = torch.randint(0, 100, (50, 16))
    return DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=10,
    )


@pytest.fixture
def train_loader():
    """Create training dataset."""
    input_ids = torch.randint(0, 100, (200, 16))
    labels = torch.randint(0, 100, (200, 16))
    return DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=20,
        shuffle=True,
    )


def test_e2e_single_model_trajectory(probe_loader, train_loader):
    """Test extracting trajectory from single model training."""
    # Create model
    class SimpleLMModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = TinyLM()

        def forward(self, input_ids, labels=None):
            return self.network(input_ids, labels=labels)

    from lightning import LightningModule

    class LitLM(LightningModule):
        def __init__(self):
            super().__init__()
            self.network = TinyLM()

        def forward(self, input_ids, labels=None):
            return self.network(input_ids, labels=labels)

        def training_step(self, batch, batch_idx):
            input_ids, labels = batch
            out = self(input_ids, labels=labels)
            self.log("train/loss", out.loss)
            return out.loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    model = LitLM()

    callback = RepresentationAuditCallback(
        probe_loader=probe_loader,
        layer_specs=[LayerSpec(path="network.layers[-1]", reduce="mean")],
        trigger=AuditTrigger(every_n_steps=5),
    )

    trainer = Trainer(
        max_epochs=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_loader)

    trajectory = callback.get_trajectory()

    # Should have multiple points
    assert len(trajectory) > 2

    # Visualize
    viz = TrajectoryVisualizer(n_components=2)
    embedding = viz.fit_transform(trajectory)

    assert embedding.shape[0] == len(trajectory)
    assert embedding.shape[1] == 2


@pytest.mark.slow
def test_e2e_multi_model_convergence(probe_loader, train_loader):
    """Test convergence analysis across multiple model replicas."""
    from lightning import LightningModule, seed_everything

    class LitLM(LightningModule):
        def __init__(self, seed: int):
            super().__init__()
            seed_everything(seed)
            self.network = TinyLM()

        def forward(self, input_ids, labels=None):
            return self.network(input_ids, labels=labels)

        def training_step(self, batch, batch_idx):
            input_ids, labels = batch
            out = self(input_ids, labels=labels)
            return out.loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    trajectories = []

    # Train 3 replicas with different seeds
    for seed in [42, 43, 44]:
        model = LitLM(seed=seed)

        callback = RepresentationAuditCallback(
            probe_loader=probe_loader,
            layer_specs=[LayerSpec(path="network.layers[-1]", reduce="mean")],
            trigger=AuditTrigger(every_n_steps=5),
        )

        trainer = Trainer(
            max_epochs=3,
            callbacks=[callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, train_loader)
        trajectories.append(callback.get_trajectory())

    # Compute spread over time
    spreads = compute_multi_model_spread(trajectories)

    assert len(spreads) > 0
    # Note: With random init and short training, convergence isn't guaranteed
    # This test just verifies the pipeline works
```

**Step 2: Run E2E test**

Run: `pytest manylatents/tests/integration/test_convergent_trajectories.py -v`
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add manylatents/tests/integration/test_convergent_trajectories.py
git commit -m "test(integration): add E2E test for convergent trajectories experiment"
```

---

### Task 22: E2E test for merging experiment

**Files:**
- Create: `manylatents/tests/integration/test_merging_experiment.py`

**Step 1: Write E2E test for merging workflow**

```python
# manylatents/tests/integration/test_merging_experiment.py
"""End-to-end test for representation merging experiment."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from manylatents.lightning.hooks import ActivationExtractor, LayerSpec
from manylatents.gauge.diffusion import DiffusionGauge
from manylatents.algorithms.latent.merging import DiffusionMerging
from manylatents.algorithms.latent.phate import PHATEModule


class TinyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=64, hidden_dim=32, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def probe_data():
    """Shared probe dataset."""
    torch.manual_seed(42)
    return torch.randn(100, 64)


def test_e2e_merging_workflow(probe_data):
    """Test full merging workflow: extract → gauge → merge → PHATE."""
    # Step 1: Create multiple "pretrained" models
    models = {
        "model_a": TinyModel(seed=1),
        "model_b": TinyModel(seed=2),
        "model_c": TinyModel(seed=3),
    }

    # Step 2: Extract activations from each model on probe set
    spec = LayerSpec(path="layers.2", reduce="none")  # Last layer
    gauge = DiffusionGauge(knn=10)

    diffusion_ops = {}
    for name, model in models.items():
        extractor = ActivationExtractor([spec])

        model.eval()
        with torch.no_grad():
            with extractor.capture(model):
                _ = model(probe_data)

        activations = extractor.get_activations()
        diff_op = gauge(activations["layers.2"])
        diffusion_ops[name] = diff_op

    # Step 3: Merge diffusion operators
    merger = DiffusionMerging(strategy="frobenius_mean")
    merged_op = merger.merge(diffusion_ops)

    assert merged_op.shape == (100, 100)

    # Step 4: Embed merged operator with PHATE
    # Convert operator to features (use rows as features)
    phate = PHATEModule(n_components=2, knn=10, t=5)
    phate.fit(torch.from_numpy(merged_op).float())
    embedding = phate.transform(torch.from_numpy(merged_op).float())

    assert embedding.shape == (100, 2)


def test_e2e_student_target_loss(probe_data):
    """Test computing loss between student embedding and target."""
    # Create target embedding via merge
    models = {
        "model_a": TinyModel(seed=1),
        "model_b": TinyModel(seed=2),
    }

    spec = LayerSpec(path="layers.2", reduce="none")
    gauge = DiffusionGauge(knn=10)

    diffusion_ops = {}
    for name, model in models.items():
        extractor = ActivationExtractor([spec])
        model.eval()
        with torch.no_grad():
            with extractor.capture(model):
                _ = model(probe_data)
        activations = extractor.get_activations()
        diffusion_ops[name] = gauge(activations["layers.2"])

    merger = DiffusionMerging(strategy="weighted_interpolation")
    merged_op = merger.merge(diffusion_ops)

    # PHATE embedding of merged operator = target
    phate = PHATEModule(n_components=2, knn=10, t=5)
    target_embedding = phate.fit_transform(torch.from_numpy(merged_op).float())

    # Student model
    student = TinyModel(seed=99)

    # Get student activations
    extractor = ActivationExtractor([spec])
    with extractor.capture(student):
        _ = student(probe_data)
    student_acts = extractor.get_activations()["layers.2"]

    # Student diffusion operator
    student_op = gauge(student_acts)
    student_embedding = phate.transform(torch.from_numpy(student_op).float())

    # Compute alignment loss (e.g., MSE between embeddings)
    if isinstance(target_embedding, torch.Tensor):
        target_embedding = target_embedding.numpy()
    if isinstance(student_embedding, torch.Tensor):
        student_embedding = student_embedding.numpy()

    alignment_loss = np.mean((student_embedding - target_embedding) ** 2)

    # Should be positive (student != target initially)
    assert alignment_loss > 0
```

**Step 2: Run test**

Run: `pytest manylatents/tests/integration/test_merging_experiment.py -v`
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add manylatents/tests/integration/test_merging_experiment.py
git commit -m "test(integration): add E2E test for merging experiment workflow"
```

---

### Task 23: Final documentation and module exports

**Files:**
- Modify: `manylatents/lightning/__init__.py`
- Modify: `manylatents/gauge/__init__.py`
- Create: `docs/guide/representation-audit.md`

**Step 1: Update module exports**

```python
# manylatents/lightning/__init__.py
"""PyTorch Lightning modules and callbacks for manylatents."""
from manylatents.lightning.hooks import (
    LayerSpec,
    ActivationExtractor,
    resolve_layer,
)
from manylatents.lightning.hf_trainer import (
    HFTrainerModule,
    HFTrainerConfig,
)
from manylatents.lightning.callbacks import (
    AuditTrigger,
    RepresentationAuditCallback,
)

__all__ = [
    "LayerSpec",
    "ActivationExtractor",
    "resolve_layer",
    "HFTrainerModule",
    "HFTrainerConfig",
    "AuditTrigger",
    "RepresentationAuditCallback",
]
```

```python
# manylatents/gauge/__init__.py
"""Gauge modules for computing geometric operators from representations."""
from manylatents.gauge.diffusion import DiffusionGauge
from manylatents.gauge.trajectory import (
    TrajectoryVisualizer,
    compute_multi_model_spread,
)

__all__ = [
    "DiffusionGauge",
    "TrajectoryVisualizer",
    "compute_multi_model_spread",
]
```

**Step 2: Write documentation**

```markdown
# docs/guide/representation-audit.md
# Representation Auditing Guide

This guide covers how to audit neural network representations during training using manylatents.

## Overview

The representation audit system allows you to:
- Extract activations from specific layers during training
- Compute diffusion operators from activations
- Track representation trajectories over training
- Merge representations across multiple models
- Visualize convergence in diffusion space

## Quick Start

```python
from manylatents.lightning import (
    HFTrainerModule,
    HFTrainerConfig,
    RepresentationAuditCallback,
    AuditTrigger,
    LayerSpec,
)
from torch.utils.data import DataLoader

# Configure model
config = HFTrainerConfig(model_name_or_path="gpt2")
module = HFTrainerModule(config)

# Create probe loader (fixed dataset for auditing)
probe_loader = DataLoader(...)

# Configure audit callback
callback = RepresentationAuditCallback(
    probe_loader=probe_loader,
    layer_specs=[
        LayerSpec(path="transformer.h[-1]", reduce="mean"),
    ],
    trigger=AuditTrigger(every_n_steps=100),
    log_to_wandb=True,
)

# Train with auditing
trainer = Trainer(callbacks=[callback])
trainer.fit(module, train_loader)

# Get trajectory
trajectory = callback.get_trajectory()
```

## Layer Specification

```python
LayerSpec(
    path="model.layers[-1]",     # Dot-path to layer
    extraction_point="output",    # "output", "input", "hidden_states"
    reduce="mean",                # "mean", "last_token", "cls", "all"
)
```

## Diffusion Gauge

```python
from manylatents.gauge import DiffusionGauge

gauge = DiffusionGauge(
    knn=15,           # Neighbors for adaptive bandwidth
    alpha=1.0,        # Diffusion normalization
    symmetric=True,   # Symmetric operator
)

diff_op = gauge(activations)  # (N, N) diffusion operator
```

## Merging Operators

```python
from manylatents.algorithms.latent.merging import DiffusionMerging

merger = DiffusionMerging(strategy="frobenius_mean")
merged_op = merger.merge({
    "model_a": op_a,
    "model_b": op_b,
})
```

Strategies:
- `weighted_interpolation`: Weighted sum, normalized
- `frobenius_mean`: Arithmetic mean
- `ot_barycenter`: Wasserstein barycenter (requires POT)

## Trajectory Visualization

```python
from manylatents.gauge import TrajectoryVisualizer, compute_multi_model_spread

# Embed trajectory
viz = TrajectoryVisualizer(n_components=2)
embedding = viz.fit_transform(trajectory)

# Multi-model convergence
spreads = compute_multi_model_spread(trajectories)
```

## Parallel Runs with Shop

See `shop/configs/experiments/representation_audit_sweep.yaml` for SLURM sweep configuration.

```bash
python shop/scripts/launch_audit_sweep.py --cluster mila
```
```

**Step 3: Commit**

```bash
git add manylatents/lightning/__init__.py manylatents/gauge/__init__.py docs/guide/representation-audit.md
git commit -m "docs: add representation audit guide and update module exports"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|------------------|
| 1. Core Extraction | 1-4 | LayerSpec, resolve_layer, ActivationExtractor |
| 2. DiffusionGauge | 5-6 | DiffusionGauge with kernel options |
| 3. HFTrainerModule | 7-8 | Lightning wrapper for HF models |
| 4. AuditCallback | 9-11 | RepresentationAuditCallback with triggers |
| 5. DiffusionMerging | 12-13 | Operator merging strategies |
| 6. Visualization | 14-15 | TrajectoryVisualizer, spread metrics |
| 7. WandB | 16-17 | WandbAuditLogger integration |
| 8. Shop/SLURM | 18-20 | Hydra configs, launchers, aggregation |
| 9. Integration | 21-23 | E2E tests, docs, exports |

**Total: 23 tasks across 9 phases**
