# manylatents/algorithms/latent/tests/test_diffusion_merging.py
import sys
import pytest
import numpy as np
from manylatents.algorithms.latent.merging import DiffusionMerging

# Check if POT is available
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False


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
    # Frobenius mean is arithmetic mean (before normalization)
    expected = (ops["model_a"] + ops["model_b"] + ops["model_c"]) / 3
    # Normalize expected
    row_sums = expected.sum(axis=1, keepdims=True)
    expected = expected / row_sums
    np.testing.assert_allclose(merged, expected, rtol=1e-5)


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


def test_diffusion_merging_invalid_strategy():
    """Should raise on invalid strategy."""
    with pytest.raises(ValueError, match="strategy must be one of"):
        DiffusionMerging(strategy="invalid")


def test_diffusion_merging_empty_operators():
    """Should raise on empty operators dict."""
    merger = DiffusionMerging()
    with pytest.raises(ValueError, match="operators dict is empty"):
        merger.merge({})


@pytest.mark.skipif(not HAS_POT, reason="POT not installed")
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
    # Temporarily hide ot module to test error path
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
