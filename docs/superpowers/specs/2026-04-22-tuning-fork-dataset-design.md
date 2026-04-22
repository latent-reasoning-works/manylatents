# Tuning Fork Synthetic Dataset — Design Spec

**Date:** 2026-04-22
**Branch:** `dataset/tuning-fork`

## Goal

A synthetic manifold dataset shaped like a tuning fork, with intentionally heterogeneous density. Designed to expose the failure modes of global neighborhood selection in manifold learning methods (PHATE, UMAP, t-SNE, etc.):

- **Too small a neighborhood** → handle fragments (sparse region disconnects)
- **Too large a neighborhood** → prongs mix (close in Euclidean space, far on manifold)

## Geometry

A 1D manifold embedded in 2D (optionally rotated into higher D), consisting of three arcs:

1. **Handle** — vertical segment from `(0, 0)` to `(0, handle_length)`
2. **Left arm** — quarter-circle bend from junction curving left, then straight prong running upward at `x = −dist_between_prongs/2`
3. **Right arm** — mirror of left arm at `x = +dist_between_prongs/2`

The bend radius equals `dist_between_prongs / 2`, making the transition geometrically smooth. Prongs run parallel for their full length (U-shape), so Euclidean proximity between prongs is uniform — cross-prong shortcuts can occur anywhere along the prong length.

Points are sampled **uniformly by arc length** on each arm. The bend is included in the prong arc and gets proportional density — no gap at the junction.

**Labels:**
- `0` — handle
- `1` — left arm (bend + straight prong)
- `2` — right arm (bend + straight prong)

## Parameters

| Param | Default | Description |
|---|---|---|
| `n_prong` | `500` | Points per prong arm (bend + straight section) |
| `handle_prong_ratio` | `0.2` | Handle gets `n_handle = int(handle_prong_ratio × n_prong)` points. Total N = `n_handle + 2 × n_prong`. |
| `dist_between_prongs` | `0.3` | Euclidean gap between the two parallel prong lines. Keep small so k-NN crosses prongs. |
| `prong_length` | `3.0` | Length of the straight prong section |
| `handle_length` | `2.0` | Length of the handle segment |
| `noise` | `0.02` | Gaussian observation noise std added to all points |
| `rotate_to_dim` | `2` | If > 2, randomly rotate data into this many dimensions (like SwissRoll) |
| `random_state` | `42` | RNG seed |
| `save_viz` | `False` | If True, save ground truth PNG on instantiation |
| `save_dir` | `"outputs"` | Directory for saved PNG |

## Ground Truth

- **`get_gt_dists()`** — returns `(N, N)` pairwise arc-length geodesic distances. Each point has a 1D arc-length coordinate (cumulative arc length from handle bottom); distances are pairwise absolute differences in this coordinate.
- **Visualization** (`save_viz=True`) — matplotlib PNG saved to `save_dir/tuning_fork_ground_truth.png`: gray skeleton curve overlaid with scatter of points colored by label (blue=handle, green=left arm, red=right arm). Density is directly visible from point concentration.

## Three Dataset Variants (YAML configs)

### 1. Positive case — `tuning_fork_positive.yaml`
Sparse handle, dense prongs. Main regime of interest.

```yaml
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
```

Expected behavior:
- Small neighborhood → handle fragments
- Large neighborhood → prongs mix
- Local adaptation may help

### 2. Dense control — `tuning_fork_dense_control.yaml`
Dense handle, dense prongs. Little benefit expected from local adaptation.

```yaml
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
```

Expected behavior: small global neighborhood already works well.

### 3. Sparse control — `tuning_fork_sparse_control.yaml`
Sparse everywhere. No good global or local solution.

```yaml
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
```

Expected behavior: fragmentation everywhere, no neighborhood size is clearly good.

## Files

| File | Action |
|---|---|
| `manylatents/data/synthetic_dataset.py` | Add `TuningFork(SyntheticDataset)` class |
| `manylatents/data/tuning_fork.py` | Add `TuningForkDataModule(LightningDataModule)` |
| `manylatents/data/__init__.py` | Auto-discovered; no changes needed |
| `manylatents/configs/data/tuning_fork_positive.yaml` | New config |
| `manylatents/configs/data/tuning_fork_dense_control.yaml` | New config |
| `manylatents/configs/data/tuning_fork_sparse_control.yaml` | New config |
| `tests/test_tuning_fork.py` | New test file |

## Tests

- Output shape `(N, 2)` where `N = n_handle + 2 × n_prong`
- Exactly 3 unique label values (0, 1, 2)
- `rotate_to_dim=50` → shape `(N, 50)`
- `get_gt_dists()` returns `(N, N)` symmetric matrix
- Determinism: same `random_state` → identical output
- DataModule integration: Hydra can instantiate each of the 3 configs
