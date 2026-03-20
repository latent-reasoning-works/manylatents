"""End-to-end: the full direct Python API workflow."""
import numpy as np
import pytest


def test_full_workflow_ndarray_evaluate_cache():
    """Complete workflow: ndarray fit -> evaluate_metrics -> shared cache."""
    from manylatents.algorithms.latent.pca import PCAModule
    from manylatents import evaluate_metrics

    rng = np.random.RandomState(42)
    X = rng.randn(100, 10).astype(np.float32)
    cache = {}

    # Fit with ndarray
    mod = PCAModule(n_components=2)
    emb = mod.fit_transform(X)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (100, 2)

    # Evaluate metrics
    scores = evaluate_metrics(emb, metrics=["FractalDimension"], cache=cache)
    assert isinstance(scores, dict)
    assert "FractalDimension" in scores

    # Second module, same cache
    mod2 = PCAModule(n_components=5)
    emb2 = mod2.fit_transform(X)
    scores2 = evaluate_metrics(emb2, metrics=["FractalDimension"], cache=cache)
    assert "FractalDimension" in scores2
