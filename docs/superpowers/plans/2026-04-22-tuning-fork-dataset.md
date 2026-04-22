# Tuning Fork Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TuningFork` synthetic dataset — a U-shaped manifold with heterogeneous density — to expose global neighborhood failures in manifold learning.

**Architecture:** `TuningFork(SyntheticDataset)` in `synthetic_dataset.py` handles geometry/sampling; `TuningForkDataModule(LightningDataModule)` in `tuning_fork.py` wraps it for Lightning; three Hydra YAML configs cover the positive case and two controls. Follows the exact same patterns as `Archetypal`/`SwissRoll`.

**Tech Stack:** NumPy, SciPy (`special_ortho_group`), Matplotlib (lazy import, `Agg` backend), PyTorch Lightning, Hydra.

---

## File Map

| File | Action |
|---|---|
| `manylatents/data/synthetic_dataset.py` | Add `TuningFork` class at end of file |
| `manylatents/data/tuning_fork.py` | Create — `TuningForkDataModule` |
| `manylatents/configs/data/tuning_fork_positive.yaml` | Create |
| `manylatents/configs/data/tuning_fork_dense_control.yaml` | Create |
| `manylatents/configs/data/tuning_fork_sparse_control.yaml` | Create |
| `tests/test_tuning_fork.py` | Create |

---

## Task 1: TuningFork dataset — geometry and sampling

**Files:**
- Modify: `manylatents/data/synthetic_dataset.py` (append at end)
- Create: `tests/test_tuning_fork.py`

### Geometry reference

The 2D skeleton has three arcs. With `half_gap = dist_between_prongs / 2`:

- **Handle**: `(0, 0)` → `(0, handle_length)` (vertical, arc length = `handle_length`)
- **Left bend**: quarter-circle, center `(-half_gap, handle_length)`, radius `half_gap`, θ ∈ [0, π/2]
  - `x(θ) = -half_gap + half_gap·cos(θ)`, `y(θ) = handle_length + half_gap·sin(θ)`
  - arc length = `π/2 · half_gap`
- **Left straight**: `(-half_gap, handle_length + half_gap)` → `(-half_gap, handle_length + half_gap + prong_length)`
- **Right bend**: mirror of left (`+half_gap`, with `cos` term negated)
  - `x(θ) = half_gap - half_gap·cos(θ)`, `y(θ) = handle_length + half_gap·sin(θ)`
- **Right straight**: mirror of left straight

Points are sampled uniformly by arc length within each arc. `n_handle = int(handle_prong_ratio * n_prong)`.

- [ ] **Step 1: Write failing tests for geometry and sampling**

```python
# tests/test_tuning_fork.py
import numpy as np
import pytest
from manylatents.data.synthetic_dataset import TuningFork, SyntheticDataset


def _make(n_prong=50, **kw):
    return TuningFork(n_prong=n_prong, random_state=0, **kw)


def test_isinstance_synthetic_dataset():
    assert isinstance(_make(), SyntheticDataset)


def test_shape_default():
    ds = _make(n_prong=50, handle_prong_ratio=0.2)
    n_handle = int(0.2 * 50)
    assert ds.data.shape == (n_handle + 2 * 50, 2)


def test_shape_total_n():
    ds = _make(n_prong=100, handle_prong_ratio=0.5)
    n_handle = int(0.5 * 100)
    assert ds.data.shape == (n_handle + 200, 2)


def test_labels_three_values():
    ds = _make(n_prong=60, handle_prong_ratio=0.3)
    assert set(ds.metadata.tolist()) == {0, 1, 2}


def test_label_counts():
    n_prong = 80
    ratio = 0.25
    ds = TuningFork(n_prong=n_prong, handle_prong_ratio=ratio, random_state=0)
    n_handle = int(ratio * n_prong)
    assert np.sum(ds.metadata == 0) == n_handle
    assert np.sum(ds.metadata == 1) == n_prong
    assert np.sum(ds.metadata == 2) == n_prong


def test_determinism():
    ds1 = TuningFork(n_prong=50, random_state=7)
    ds2 = TuningFork(n_prong=50, random_state=7)
    np.testing.assert_array_equal(ds1.data, ds2.data)
    np.testing.assert_array_equal(ds1.metadata, ds2.metadata)


def test_different_seeds_differ():
    ds1 = TuningFork(n_prong=50, random_state=1)
    ds2 = TuningFork(n_prong=50, random_state=2)
    assert not np.array_equal(ds1.data, ds2.data)


def test_len():
    ds = _make(n_prong=50, handle_prong_ratio=0.4)
    n_handle = int(0.4 * 50)
    assert len(ds) == n_handle + 100


def test_getitem_keys():
    ds = _make()
    item = ds[0]
    assert "data" in item and "metadata" in item


def test_prongs_close_in_euclidean():
    # With small dist_between_prongs, facing prong points should be close
    ds = TuningFork(n_prong=200, dist_between_prongs=0.1, prong_length=3.0,
                    noise=0.0, random_state=0)
    left = ds.data[ds.metadata == 1]
    right = ds.data[ds.metadata == 2]
    # Min Euclidean distance between any left/right pair should be ~dist_between_prongs
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(left[:10], right[:10])
    assert dists.min() < 0.5  # much less than prong_length


def test_rotate_to_dim():
    n_prong = 50
    ratio = 0.2
    ds = TuningFork(n_prong=n_prong, handle_prong_ratio=ratio,
                    rotate_to_dim=50, random_state=0)
    n_handle = int(ratio * n_prong)
    assert ds.data.shape == (n_handle + 2 * n_prong, 50)


def test_handle_sparser_than_prongs():
    ds = TuningFork(n_prong=100, handle_prong_ratio=0.2, prong_length=3.0,
                    handle_length=2.0, noise=0.0, random_state=0)
    n_handle = np.sum(ds.metadata == 0)
    n_prong = np.sum(ds.metadata == 1)
    assert n_handle < n_prong
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_tuning_fork.py -x -q 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError` — `TuningFork` does not exist yet.

- [ ] **Step 3: Implement `TuningFork` class — append to `synthetic_dataset.py`**

Add at the end of `manylatents/data/synthetic_dataset.py`:

```python
class TuningFork(SyntheticDataset):
    """
    Synthetic tuning-fork manifold with heterogeneous density.

    Geometry: a 1D manifold embedded in 2D (optionally rotated to higher D)
    consisting of a sparse handle and two dense parallel prongs (U-shape).
    The prongs are close in Euclidean space so large neighborhoods create
    spurious cross-prong connections in manifold learning methods.

    Labels: 0 = handle, 1 = left prong (incl. bend), 2 = right prong (incl. bend).
    """

    def __init__(
        self,
        n_prong: int = 500,
        handle_prong_ratio: float = 0.2,
        dist_between_prongs: float = 0.3,
        prong_length: float = 3.0,
        handle_length: float = 2.0,
        noise: float = 0.02,
        rotate_to_dim: int = 2,
        random_state: int = 42,
        save_viz: bool = False,
        save_dir: str = "outputs",
    ):
        super().__init__()
        np.random.seed(random_state)
        rng = np.random.default_rng(random_state)

        self.save_viz = save_viz
        self.save_dir = save_dir

        half_gap = dist_between_prongs / 2.0
        bend_arc = np.pi / 2.0 * half_gap
        arm_arc = bend_arc + prong_length  # total arc length per arm
        n_handle = int(handle_prong_ratio * n_prong)

        # --- Sample handle points uniformly along arc length ---
        # Handle runs from y=0 to y=handle_length at x=0
        h_pos = rng.uniform(0.0, handle_length, size=n_handle)
        handle_pts = np.stack([np.zeros(n_handle), h_pos], axis=1)

        # --- Sample arm points uniformly along arc length ---
        def _sample_arm(n, sign):
            """sign=+1 for right arm, sign=-1 for left arm."""
            s = rng.uniform(0.0, arm_arc, size=n)
            x = np.empty(n)
            y = np.empty(n)
            on_bend = s < bend_arc
            # Bend: quarter-circle
            theta = s[on_bend] / half_gap  # angle in [0, pi/2]
            x[on_bend] = sign * half_gap - sign * half_gap * np.cos(theta)
            y[on_bend] = handle_length + half_gap * np.sin(theta)
            # Straight prong
            straight_s = s[~on_bend] - bend_arc
            x[~on_bend] = sign * half_gap
            y[~on_bend] = handle_length + half_gap + straight_s
            return np.stack([x, y], axis=1)

        left_pts = _sample_arm(n_prong, sign=-1)
        right_pts = _sample_arm(n_prong, sign=+1)

        # Store arc-length section/position for get_gt_dists
        self._handle_length = handle_length
        self._arc_section = np.concatenate([
            np.zeros(n_handle, dtype=int),
            np.ones(n_prong, dtype=int),
            np.full(n_prong, 2, dtype=int),
        ])

        def _arm_arc_pos(n, sign):
            s = rng.uniform(0.0, arm_arc, size=n)
            # we already sampled via rng above; store positions from the same draws
            # NOTE: positions stored separately in _arc_pos are from a fresh draw;
            # to be consistent we track s directly from the samples above.
            return s

        # Re-derive arc positions from the already-sampled points
        # (avoid double-sampling: compute arc pos from geometry)
        def _arc_pos_from_pts(pts, section, half_gap, handle_length, bend_arc):
            if section == 0:
                return pts[:, 1]  # y coordinate = arc position from bottom
            else:
                # Determine if on bend or straight
                on_prong_straight = pts[:, 1] > (handle_length + half_gap)
                pos = np.empty(len(pts))
                # Straight: arc pos = bend_arc + (y - (handle_length + half_gap))
                pos[on_prong_straight] = (
                    bend_arc + pts[on_prong_straight, 1] - (handle_length + half_gap)
                )
                # Bend: arc pos = half_gap * arcsin((y - handle_length) / half_gap)
                # (since y = handle_length + half_gap * sin(theta), theta = s/half_gap)
                bend_mask = ~on_prong_straight
                if bend_mask.any():
                    sin_theta = (pts[bend_mask, 1] - handle_length) / half_gap
                    sin_theta = np.clip(sin_theta, -1.0, 1.0)
                    pos[bend_mask] = half_gap * np.arcsin(sin_theta)
                return pos

        self._arc_pos = np.concatenate([
            _arc_pos_from_pts(handle_pts, 0, half_gap, handle_length, bend_arc),
            _arc_pos_from_pts(left_pts, 1, half_gap, handle_length, bend_arc),
            _arc_pos_from_pts(right_pts, 2, half_gap, handle_length, bend_arc),
        ])

        data = np.concatenate([handle_pts, left_pts, right_pts], axis=0)
        labels = np.concatenate([
            np.zeros(n_handle, dtype=int),
            np.ones(n_prong, dtype=int),
            np.full(n_prong, 2, dtype=int),
        ])

        # Add isotropic Gaussian noise
        data = data + rng.normal(0.0, noise, size=data.shape)

        self.data = data
        self.metadata = labels

        if rotate_to_dim > 2:
            self.data = self.rotate_to_dim(rotate_to_dim)

        if save_viz:
            self._save_figure()
```

- [ ] **Step 4: Run tests — expect most to pass (gt_dists and viz come later)**

```bash
uv run pytest tests/test_tuning_fork.py -x -q 2>&1 | tail -15
```

Expected: geometry/shape/label/determinism tests pass. `get_gt_dists`-related tests not yet written.

- [ ] **Step 5: Commit**

```bash
git add manylatents/data/synthetic_dataset.py tests/test_tuning_fork.py
git commit -m "feat: TuningFork dataset — geometry, sampling, labels"
```

---

## Task 2: Ground truth distances

**Files:**
- Modify: `manylatents/data/synthetic_dataset.py` — add `get_gt_dists` to `TuningFork`
- Modify: `tests/test_tuning_fork.py` — add gt_dists tests

- [ ] **Step 1: Add gt_dists tests**

Append to `tests/test_tuning_fork.py`:

```python
def test_gt_dists_shape():
    ds = _make(n_prong=40, handle_prong_ratio=0.25)
    n = len(ds)
    D = ds.get_gt_dists()
    assert D.shape == (n, n)


def test_gt_dists_symmetric():
    ds = _make(n_prong=40)
    D = ds.get_gt_dists()
    np.testing.assert_allclose(D, D.T, atol=1e-10)


def test_gt_dists_diagonal_zero():
    ds = _make(n_prong=40)
    D = ds.get_gt_dists()
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)


def test_gt_dists_cross_prong_larger_than_same_prong():
    # Geodesic distance between prongs goes through junction;
    # should be larger than within-prong distances at the same arc depth.
    ds = TuningFork(n_prong=200, dist_between_prongs=0.1, noise=0.0, random_state=0)
    D = ds.get_gt_dists()
    left_idx = np.where(ds.metadata == 1)[0]
    right_idx = np.where(ds.metadata == 2)[0]
    same_prong_dists = D[left_idx[:5], :][:, left_idx[5:10]].flatten()
    cross_prong_dists = D[left_idx[:5], :][:, right_idx[:5]].flatten()
    assert cross_prong_dists.mean() > same_prong_dists.mean()
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/test_tuning_fork.py::test_gt_dists_shape -x -q 2>&1 | tail -10
```

Expected: `TypeError` — `get_gt_dists` returns `None`.

- [ ] **Step 3: Implement `get_gt_dists` — add method to `TuningFork` class**

Add after `__init__` in the `TuningFork` class:

```python
    def get_gt_dists(self):
        """Pairwise geodesic distances along the tuning-fork manifold.

        Geodesic paths go through the junction for cross-arm pairs.
        Same-section pairs use direct arc-length difference.
        """
        sec = self._arc_section  # (n,) — 0=handle, 1=left, 2=right
        pos = self._arc_pos      # (n,) — arc-length within section

        # Distance from each point to the junction
        # Handle: junction is at arc-pos == self._handle_length
        # Arms: junction is at arc-pos == 0
        dist_to_junc = np.where(sec == 0, self._handle_length - pos, pos)

        same_section = sec[:, None] == sec[None, :]  # (n, n)
        D = np.where(
            same_section,
            np.abs(pos[:, None] - pos[None, :]),
            dist_to_junc[:, None] + dist_to_junc[None, :],
        )
        return D
```

- [ ] **Step 4: Run all gt_dists tests**

```bash
uv run pytest tests/test_tuning_fork.py -k "gt_dists" -v 2>&1 | tail -15
```

Expected: all 4 pass.

- [ ] **Step 5: Run full test file**

```bash
uv run pytest tests/test_tuning_fork.py -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add manylatents/data/synthetic_dataset.py tests/test_tuning_fork.py
git commit -m "feat: TuningFork.get_gt_dists — arc-length geodesic distances"
```

---

## Task 3: Ground truth visualization

**Files:**
- Modify: `manylatents/data/synthetic_dataset.py` — add `_save_figure` to `TuningFork`
- Modify: `tests/test_tuning_fork.py` — add viz tests

- [ ] **Step 1: Add visualization tests**

Append to `tests/test_tuning_fork.py`:

```python
def test_save_viz_default_false(tmp_path):
    TuningFork(n_prong=30, random_state=0, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_save_viz_creates_png(tmp_path):
    TuningFork(n_prong=30, random_state=0, save_viz=True, save_dir=str(tmp_path))
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/test_tuning_fork.py::test_save_viz_creates_png -x -q 2>&1 | tail -10
```

Expected: FAIL — `_save_figure` not yet implemented.

- [ ] **Step 3: Implement `_save_figure` — add method to `TuningFork` class**

Add after `get_gt_dists` in the `TuningFork` class:

```python
    def _save_figure(self):
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.save_dir, exist_ok=True)

        colors = {0: "#4c96e8", 1: "#3fb950", 2: "#f78166"}
        labels_str = {0: "handle", 1: "left prong", 2: "right prong"}

        fig, ax = plt.subplots(figsize=(4, 8))

        # Draw skeleton in 2D (use self.data only if not rotated)
        # Reconstruct skeleton from stored arc metadata — always in 2D
        half_gap = None
        # Infer geometry from stored arc positions for handle/arm extents
        handle_pts = self.data[self._arc_section == 0] if self.data.shape[1] == 2 else None
        left_pts = self.data[self._arc_section == 1] if self.data.shape[1] == 2 else None
        right_pts = self.data[self._arc_section == 2] if self.data.shape[1] == 2 else None

        for label, pts in [(0, handle_pts), (1, left_pts), (2, right_pts)]:
            if pts is not None and len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=6, c=colors[label],
                           label=labels_str[label], alpha=0.7, linewidths=0)

        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Tuning Fork — Ground Truth")
        ax.set_xlabel("x")
        ax.set_ylabel("y (arc)")

        path = os.path.join(self.save_dir, "tuning_fork_ground_truth.png")
        plt.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
```

- [ ] **Step 4: Run visualization tests**

```bash
uv run pytest tests/test_tuning_fork.py -k "save_viz" -v 2>&1 | tail -10
```

Expected: both pass.

- [ ] **Step 5: Run full test file**

```bash
uv run pytest tests/test_tuning_fork.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add manylatents/data/synthetic_dataset.py tests/test_tuning_fork.py
git commit -m "feat: TuningFork ground truth visualization"
```

---

## Task 4: TuningForkDataModule and YAML configs

**Files:**
- Create: `manylatents/data/tuning_fork.py`
- Create: `manylatents/configs/data/tuning_fork_positive.yaml`
- Create: `manylatents/configs/data/tuning_fork_dense_control.yaml`
- Create: `manylatents/configs/data/tuning_fork_sparse_control.yaml`
- Modify: `tests/test_tuning_fork.py` — add DataModule tests

- [ ] **Step 1: Add DataModule tests**

Append to `tests/test_tuning_fork.py`:

```python
from manylatents.data.tuning_fork import TuningForkDataModule


def test_datamodule_setup_full():
    dm = TuningForkDataModule(n_prong=50, mode="full", random_state=0)
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.test_dataset is not None


def test_datamodule_setup_split():
    dm = TuningForkDataModule(n_prong=100, handle_prong_ratio=0.2,
                               mode="split", test_split=0.2, random_state=0)
    dm.setup()
    n_handle = int(0.2 * 100)
    total = n_handle + 200
    assert len(dm.train_dataset) + len(dm.test_dataset) == total


def test_datamodule_train_dataloader():
    dm = TuningForkDataModule(n_prong=50, batch_size=16, mode="full", random_state=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "data" in batch and "metadata" in batch


def test_datamodule_save_viz_default_false(tmp_path):
    dm = TuningForkDataModule(n_prong=30, random_state=0, save_dir=str(tmp_path))
    dm.setup()
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run DataModule tests to confirm they fail**

```bash
uv run pytest tests/test_tuning_fork.py::test_datamodule_setup_full -x -q 2>&1 | tail -10
```

Expected: `ModuleNotFoundError` — `tuning_fork.py` does not exist.

- [ ] **Step 3: Create `manylatents/data/tuning_fork.py`**

```python
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .synthetic_dataset import TuningFork


class TuningForkDataModule(LightningDataModule):
    """LightningDataModule for the TuningFork synthetic dataset."""

    def __init__(
        self,
        n_prong: int = 500,
        handle_prong_ratio: float = 0.2,
        dist_between_prongs: float = 0.3,
        prong_length: float = 3.0,
        handle_length: float = 2.0,
        noise: float = 0.02,
        rotate_to_dim: int = 2,
        random_state: int = 42,
        save_viz: bool = False,
        save_dir: str = "outputs",
        batch_size: int = 128,
        num_workers: int = 0,
        shuffle_traindata: bool = True,
        test_split: float = 0.2,
        mode: str = "full",
    ):
        super().__init__()
        self.n_prong = n_prong
        self.handle_prong_ratio = handle_prong_ratio
        self.dist_between_prongs = dist_between_prongs
        self.prong_length = prong_length
        self.handle_length = handle_length
        self.noise = noise
        self.rotate_to_dim = rotate_to_dim
        self.random_state = random_state
        self.save_viz = save_viz
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.test_split = test_split
        self.mode = mode

        self.train_dataset = None
        self.test_dataset = None

    def _make_dataset(self):
        return TuningFork(
            n_prong=self.n_prong,
            handle_prong_ratio=self.handle_prong_ratio,
            dist_between_prongs=self.dist_between_prongs,
            prong_length=self.prong_length,
            handle_length=self.handle_length,
            noise=self.noise,
            rotate_to_dim=self.rotate_to_dim,
            random_state=self.random_state,
            save_viz=self.save_viz,
            save_dir=self.save_dir,
        )

    def setup(self, stage=None):
        if self.mode == "full":
            ds = self._make_dataset()
            self.train_dataset = ds
            self.test_dataset = ds
        elif self.mode == "split":
            ds = self._make_dataset()
            test_size = int(len(ds) * self.test_split)
            train_size = len(ds) - test_size
            self.train_dataset, self.test_dataset = random_split(
                ds,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle_traindata, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
```

- [ ] **Step 4: Create the three YAML configs**

`manylatents/configs/data/tuning_fork_positive.yaml`:
```yaml
defaults:
  - default

_target_: manylatents.data.tuning_fork.TuningForkDataModule
n_prong: 500
handle_prong_ratio: 0.2
dist_between_prongs: 0.3
prong_length: 3.0
handle_length: 2.0
noise: 0.02
rotate_to_dim: 2
random_state: ${seed}
save_viz: false
save_dir: ${hydra:runtime.output_dir}
mode: full
test_split: 0.2
shuffle_traindata: false
```

`manylatents/configs/data/tuning_fork_dense_control.yaml`:
```yaml
defaults:
  - default

_target_: manylatents.data.tuning_fork.TuningForkDataModule
n_prong: 500
handle_prong_ratio: 1.0
dist_between_prongs: 0.3
prong_length: 3.0
handle_length: 2.0
noise: 0.02
rotate_to_dim: 2
random_state: ${seed}
save_viz: false
save_dir: ${hydra:runtime.output_dir}
mode: full
test_split: 0.2
shuffle_traindata: false
```

`manylatents/configs/data/tuning_fork_sparse_control.yaml`:
```yaml
defaults:
  - default

_target_: manylatents.data.tuning_fork.TuningForkDataModule
n_prong: 100
handle_prong_ratio: 0.2
dist_between_prongs: 0.3
prong_length: 3.0
handle_length: 2.0
noise: 0.02
rotate_to_dim: 2
random_state: ${seed}
save_viz: false
save_dir: ${hydra:runtime.output_dir}
mode: full
test_split: 0.2
shuffle_traindata: false
```

- [ ] **Step 5: Run DataModule tests**

```bash
uv run pytest tests/test_tuning_fork.py -k "datamodule" -v 2>&1 | tail -15
```

Expected: all 4 pass.

- [ ] **Step 6: Run full test suite for this file**

```bash
uv run pytest tests/test_tuning_fork.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add manylatents/data/tuning_fork.py \
        manylatents/configs/data/tuning_fork_positive.yaml \
        manylatents/configs/data/tuning_fork_dense_control.yaml \
        manylatents/configs/data/tuning_fork_sparse_control.yaml \
        tests/test_tuning_fork.py
git commit -m "feat: TuningForkDataModule and Hydra configs (positive, dense, sparse)"
```

---

## Task 5: Hydra config smoke test

**Files:**
- Modify: `tests/test_tuning_fork.py` — add Hydra instantiation test

- [ ] **Step 1: Add Hydra config test**

Append to `tests/test_tuning_fork.py`:

```python
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import hydra.utils
import manylatents.configs  # noqa: F401 — registers ConfigStore

CONFIGS = Path(__file__).parent.parent / "manylatents" / "configs"


def _compose(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS.resolve()), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def test_hydra_instantiate_positive():
    cfg = _compose(["data=tuning_fork_positive"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None


def test_hydra_instantiate_dense_control():
    cfg = _compose(["data=tuning_fork_dense_control"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None


def test_hydra_instantiate_sparse_control():
    cfg = _compose(["data=tuning_fork_sparse_control"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    assert dm.train_dataset is not None
```

- [ ] **Step 2: Run Hydra tests**

```bash
uv run pytest tests/test_tuning_fork.py -k "hydra" -v 2>&1 | tail -15
```

Expected: all 3 pass.

- [ ] **Step 3: Run full project test suite**

```bash
uv run pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all pass (or pre-existing failures only — none introduced by this feature).

- [ ] **Step 4: Final commit**

```bash
git add tests/test_tuning_fork.py
git commit -m "test: Hydra config smoke tests for all TuningFork variants"
```

---

## Done

All tasks complete when:
- `uv run pytest tests/test_tuning_fork.py -q` passes fully
- `uv run pytest tests/ -x -q` shows no new failures
- Three configs are instantiable via `uv run python -m manylatents.main data=tuning_fork_positive trainer.fast_dev_run=true`
