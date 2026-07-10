"""Smoke test for signal-manifold geometry v0 (LID@k + per-layer AUROC).

Synthetic-only: 2 classes x N variants x per-layer vectors. No models, no GPU.
Run with: pytest tests/metrics/test_signal_geometry.py
"""

import numpy as np

from manylatents.metrics.signal_geometry import (
    SIGNAL_LAYERS,
    make_synthetic_cohort,
    signal_manifold_geometry,
)


def test_signal_manifold_geometry_smoke():
    separable = ("rna", "splice")
    layer_vectors, labels = make_synthetic_cohort(
        n_per_class=80, dim=16, separable_layers=separable, shift=4.0, seed=0
    )

    results = signal_manifold_geometry(layer_vectors, labels, k=20)

    # One geometry record per layer, keyed by the shared #26 schema layers.
    assert set(results) == set(SIGNAL_LAYERS)

    for layer, g in results.items():
        assert g.n == 160 and g.dim == 16
        # LID is a finite positive intrinsic-dimension estimate.
        assert np.isfinite(g.lid) and 0.0 < g.lid <= g.dim + 1e-6
        # Axis-aligned AUROC is always in [0.5, 1.0].
        assert 0.5 - 1e-9 <= g.auroc <= 1.0 + 1e-9

    # Layers with an injected class shift separate; pure-noise layers do not.
    sep_auroc = min(results[l].auroc for l in separable)
    noise_auroc = max(results[l].auroc for l in SIGNAL_LAYERS if l not in separable)
    assert sep_auroc > 0.9, f"separable layers should separate, got {sep_auroc:.3f}"
    assert sep_auroc > noise_auroc, (
        f"separable AUROC {sep_auroc:.3f} should beat noise AUROC {noise_auroc:.3f}"
    )


def test_cv_auroc_is_honest_on_null_layers():
    """5-fold CV pulls pure-noise layers to ~0.5 while separable layers stay high.

    The in-cohort read is optimistic (>= 0.5 by construction); the CV read fits
    the axis out-of-fold, so a null layer is free to sit at or below 0.5.
    """
    separable = ("rna", "splice")
    layer_vectors, labels = make_synthetic_cohort(
        n_per_class=80, dim=16, separable_layers=separable, shift=4.0, seed=0
    )

    cv = signal_manifold_geometry(layer_vectors, labels, k=20, cv=5)

    for layer in separable:
        assert cv[layer].auroc > 0.9
    for layer in SIGNAL_LAYERS:
        if layer not in separable:
            assert cv[layer].auroc < 0.7, f"{layer} noise AUROC(cv) too high: {cv[layer].auroc:.3f}"


def test_auroc_requires_two_classes():
    import pytest

    vectors = np.random.default_rng(1).standard_normal((40, 8))
    single_class = np.zeros(40, int)
    with pytest.raises(ValueError):
        signal_manifold_geometry({"rna": vectors}, single_class, k=20)
