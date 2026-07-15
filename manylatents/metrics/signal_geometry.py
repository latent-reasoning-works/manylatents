"""Signal-manifold geometry v0: per-layer LID + AUROC over per-variant signal vectors.

Consumes the shared #26 SignalRecord shape — a per-variant, per-layer aggregated
vector plus one binary class label per variant — and reports, for each track layer:

  * LID@k : local intrinsic dimensionality of the cohort's layer vectors
            (REUSES ``metrics/lid.py``; LID is not reimplemented here).
  * AUROC : how separable the two variant classes are along that layer's
            class-mean-difference axis (uses ``sklearn.roc_auc_score``, the same
            call ``metrics/auc.py`` already relies on).

This is the single-oracle DNA-side baseline geometry. It is modality-agnostic by
construction: the input vectors may come from AlphaGenome (DNA) or ESM (protein);
nothing below is DNA-specific. Establishing this baseline lets a second oracle's
marginal signal be measured later against the same numbers.

Runnable directly (``python -m manylatents.metrics.signal_geometry``) to print a
synthetic-cohort table; see ``tests/metrics/test_signal_geometry.py`` for the
asserted smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from manylatents.metrics.lid import LocalIntrinsicDimensionality

# Canonical dogma track layers (shared schema, owned by the #26 SignalRecord agent).
SIGNAL_LAYERS = ("accessibility", "tf", "histone", "cage", "rna", "splice")


@dataclass
class LayerGeometry:
    """Geometry summary for one track layer over a variant cohort."""

    layer: str
    lid: float  # mean LID@k across the cohort's layer vectors
    auroc: float  # class separation along the class-mean-difference axis (>= 0.5)
    n: int  # number of variants
    dim: int  # per-variant layer vector dimensionality


def _axis_projection(vectors: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project onto the class-mean-difference axis (positive class = higher)."""
    axis = vectors[y == 1].mean(axis=0) - vectors[y == 0].mean(axis=0)
    # Elementwise-then-sum rather than ``vectors @ axis``: the BLAS matmul loop
    # emits spurious FP RuntimeWarnings once torch has touched the FP control word.
    return (vectors * axis).sum(axis=1)


def _separation_auroc(
    vectors: np.ndarray, labels: np.ndarray, cv: int | None = None
) -> float:
    """AUROC of the two classes along their class-mean-difference axis.

    A cheap, modality-agnostic linear-separability score: project each variant
    onto ``mean(positive) - mean(negative)``, then score with ``roc_auc_score``.

    ``cv=None`` (default): in-cohort read — the axis is fit on the same variants
    it scores, so it is always ``>= 0.5`` and optimistic on small N.
    ``cv=k``: honest read — stratified k-fold; the axis is fit on each train
    split and scores the held-out fold, projections pooled for one AUROC. A layer
    with no class signal then sits at ~0.5 (and may dip below), which is the
    point: it stops the within-cohort optimism from masking a null layer.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(f"AUROC needs exactly 2 classes, got {classes.tolist()}")
    neg, pos = classes
    y = (labels == pos).astype(int)

    if not cv:
        return float(roc_auc_score(y, _axis_projection(vectors, y)))

    from sklearn.model_selection import StratifiedKFold

    # Clamp folds to the smallest class so StratifiedKFold can't raise on a
    # small (e.g. rare-pathogenic) class; need >= 2 per class to CV at all.
    smallest_class = int(np.bincount(y).min())
    if smallest_class < 2:
        raise ValueError(
            f"cv AUROC needs >= 2 samples per class; smallest class has {smallest_class}"
        )
    n_splits = min(cv, smallest_class)

    proj = np.empty(y.shape[0], dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_idx, test_idx in skf.split(vectors, y):
        axis = (
            vectors[train_idx][y[train_idx] == 1].mean(axis=0)
            - vectors[train_idx][y[train_idx] == 0].mean(axis=0)
        )
        # Unit-normalize per fold so out-of-fold projections pool on ONE scale:
        # un-normalized per-fold axes have different norms, which distorts the
        # ranks when projections from all folds are scored by one roc_auc_score.
        axis /= np.linalg.norm(axis) + 1e-12
        proj[test_idx] = (vectors[test_idx] * axis).sum(axis=1)
    return float(roc_auc_score(y, proj))


def layer_geometry(
    vectors: np.ndarray,
    labels: np.ndarray,
    layer: str = "",
    k: int = 20,
    cv: int | None = None,
) -> LayerGeometry:
    """LID@k + class-separation AUROC for a single layer's per-variant vectors.

    ``cv`` forwards to :func:`_separation_auroc` (``None`` = in-cohort read;
    ``k``-fold = honest held-out AUROC).
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError(f"layer '{layer}': expected (N, D) vectors, got {vectors.shape}")
    lid = float(LocalIntrinsicDimensionality(vectors, k=k))
    auroc = _separation_auroc(vectors, labels, cv=cv)
    return LayerGeometry(
        layer=layer,
        lid=lid,
        auroc=auroc,
        n=vectors.shape[0],
        dim=vectors.shape[1],
    )


def signal_manifold_geometry(
    layer_vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    k: int = 20,
    cv: int | None = None,
) -> dict[str, LayerGeometry]:
    """Per-layer LID@k + AUROC over a variant cohort's per-layer signal vectors.

    Args:
        layer_vectors: ``{layer_name -> (N, D) array}``; one row per variant.
            Layer names are expected to be a subset of :data:`SIGNAL_LAYERS`, but
            any keys are accepted (protein-side layers work identically).
        labels: ``(N,)`` binary class labels (e.g. pathogenic vs benign).
        k: neighbors for LID (default 20; matches ``metrics/lid.py``).
        cv: if set, per-layer AUROC is a stratified ``cv``-fold held-out read
            (axis fit out-of-fold, unit-normalized, projections pooled); folds
            are clamped to the smallest class. ``None`` = in-cohort read.

    Returns:
        ``{layer_name -> LayerGeometry}`` — the per-layer geometry baseline. The
        per-layer AUROC trajectory across ``SIGNAL_LAYERS`` is the eventual
        stable/building/cycling read; a per-layer number is the v0 deliverable.
    """
    labels = np.asarray(labels)
    return {
        layer: layer_geometry(vecs, labels, layer=layer, k=k, cv=cv)
        for layer, vecs in layer_vectors.items()
    }


# ---------------------------------------------------------------------------
# Synthetic demo (no models, no GPU): `python -m manylatents.metrics.signal_geometry`
# ---------------------------------------------------------------------------


def make_synthetic_cohort(
    n_per_class: int = 80,
    dim: int = 16,
    separable_layers: tuple[str, ...] = ("rna", "splice"),
    shift: float = 4.0,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two-class synthetic cohort: `separable_layers` carry class signal, rest are noise."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    labels = np.concatenate([np.zeros(n_per_class, int), np.ones(n_per_class, int)])
    layer_vectors: dict[str, np.ndarray] = {}
    for layer in SIGNAL_LAYERS:
        x = rng.standard_normal((n, dim))
        if layer in separable_layers:
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            x[labels == 1] += shift * direction
        layer_vectors[layer] = x
    return layer_vectors, labels


if __name__ == "__main__":  # pragma: no cover
    layer_vectors, labels = make_synthetic_cohort()
    results = signal_manifold_geometry(layer_vectors, labels, k=20)
    print(f"{'layer':<14}{'N':>5}{'dim':>5}{'LID@20':>10}{'AUROC':>9}")
    print("-" * 43)
    for layer in SIGNAL_LAYERS:
        g = results[layer]
        print(f"{g.layer:<14}{g.n:>5}{g.dim:>5}{g.lid:>10.3f}{g.auroc:>9.3f}")
