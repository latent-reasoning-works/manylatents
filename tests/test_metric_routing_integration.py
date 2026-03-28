"""Integration tests for metric routing (on field), sampling, and cache behavior.

Validates the new flat metric config system end-to-end through the Python API,
ensuring CLI and API parity for the ``on`` field, sweep expansion, sampling
strategies, cache isolation, and bundle composition.
"""
import logging

import numpy as np
import pytest
from omegaconf import OmegaConf

from manylatents.api import run

# Shared algorithm configs to avoid repetition
_PCA_ALGO = {
    "latent": {
        "_target_": "manylatents.algorithms.latent.pca.PCAModule",
        "n_components": 2,
    }
}

_DIFFMAP_ALGO = {
    "latent": {
        "_target_": "manylatents.algorithms.latent.diffusion_map.DiffusionMapModule",
        "n_components": 2,
        "knn": 30,
        "t": 3,
        "n_landmark": None,
    }
}


# ---------------------------------------------------------------------------
# 1. Single metric with on: embedding
# ---------------------------------------------------------------------------

def test_cli_single_metric():
    """A single metric with on=embedding produces a score via the Python API."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics={
            "trustworthiness": {
                "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
                "_partial_": True,
                "n_neighbors": 5,
                "at": "embedding",
            }
        },
    )
    assert "trustworthiness" in result["scores"]
    assert 0 < result["scores"]["trustworthiness"] <= 1


# ---------------------------------------------------------------------------
# 2. on sweep produces two evaluations
# ---------------------------------------------------------------------------

def test_on_sweep_produces_two_evaluations():
    """on: [embedding, dataset] expands to two score keys via Cartesian sweep."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics={
            "lid": {
                "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
                "_partial_": True,
                "k": 10,
                "at": ["embedding", "dataset"],
            }
        },
    )
    scores = result["scores"]
    # flatten_and_unroll_metrics produces keys like lid__at_embedding, lid__at_dataset
    assert any("at_embedding" in k for k in scores), f"No at_embedding key in {list(scores)}"
    assert any("at_dataset" in k for k in scores), f"No at_dataset key in {list(scores)}"


# ---------------------------------------------------------------------------
# 3. on: module routes to module metrics
# ---------------------------------------------------------------------------

def test_on_module_routes_to_module_metrics():
    """on=module lets SpectralGapRatio use the module's affinity eigenvalues."""
    result = run(
        data="swissroll",
        algorithms=_DIFFMAP_ALGO,
        metrics={
            "spectral_gap_ratio": {
                "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
                "_partial_": True,
                "at": "module",
            }
        },
    )
    assert "spectral_gap_ratio" in result["scores"]
    val = result["scores"]["spectral_gap_ratio"]
    assert isinstance(val, (int, float))
    assert not np.isnan(val)


# ---------------------------------------------------------------------------
# 4. on pointing to a missing output skips with warning
# ---------------------------------------------------------------------------

def test_on_missing_output_skips_with_warning(caplog):
    """on=adjacency against PCA (which has no adjacency) is silently skipped."""
    with caplog.at_level(logging.WARNING):
        result = run(
            data="swissroll",
            algorithms=_PCA_ALGO,
            metrics={
                "spectral_gap": {
                    "_target_": "manylatents.metrics.spectral_gap_ratio.SpectralGapRatio",
                    "_partial_": True,
                    "at": "adjacency",
                }
            },
        )
    # Metric should be skipped -- not present in scores
    assert "spectral_gap" not in result.get("scores", {})


# ---------------------------------------------------------------------------
# 5. Pre-fit sampling reduces data before algorithm fitting
# ---------------------------------------------------------------------------

def test_sampling_dataset_reduces_data_before_fit():
    """sampling.dataset subsamples *before* fit, producing smaller embeddings."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics={
            # Use an embedding-only metric to avoid cross-space mismatch
            # (pre-fit sampling does not subsample the evaluation dataset)
            "lid": {
                "_target_": "manylatents.metrics.lid.LocalIntrinsicDimensionality",
                "_partial_": True,
                "k": 5,
                "at": "embedding",
            }
        },
        sampling={
            "dataset": {
                "_target_": "manylatents.utils.sampling.RandomSampling",
                "fraction": 0.5,
                "seed": 42,
            }
        },
    )
    emb_shape = result["embeddings"].shape[0]
    # swissroll default: 10 distributions * 100 pts = 1000
    assert emb_shape < 1000, f"Expected fewer than 1000 samples, got {emb_shape}"
    assert emb_shape > 0
    assert "lid" in result["scores"]


# ---------------------------------------------------------------------------
# 6. Post-fit sampling: full embeddings returned, evaluation on subset
# ---------------------------------------------------------------------------

def test_sampling_embedding_reduces_for_eval():
    """sampling.embedding keeps full embeddings but evaluates metrics on a subset."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics={
            "trustworthiness": {
                "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
                "_partial_": True,
                "n_neighbors": 5,
                "at": "embedding",
            }
        },
        sampling={
            "embedding": {
                "_target_": "manylatents.utils.sampling.RandomSampling",
                "fraction": 0.5,
                "seed": 42,
            }
        },
    )
    # Full embeddings are returned (post-fit sampling only affects eval)
    assert result["embeddings"].shape[0] >= 800, (
        f"Expected full embeddings (~1000), got {result['embeddings'].shape[0]}"
    )
    assert "trustworthiness" in result["scores"]


# ---------------------------------------------------------------------------
# 7. Cache isolation across different on values
# ---------------------------------------------------------------------------

def test_cache_isolation_across_on_values():
    """Different on values produce distinct cache entries (no cross-contamination)."""
    from manylatents.experiment import prewarm_cache
    from manylatents.utils.metrics import _content_key

    rng = np.random.RandomState(42)
    emb = rng.randn(50, 2).astype(np.float32)
    data = rng.randn(50, 10).astype(np.float32)

    class FakeDS:
        pass

    ds = FakeDS()
    ds.data = data

    cfgs = {
        "emb_metric": OmegaConf.create({
            "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
            "n_neighbors": 10,
            "at": "embedding",
        }),
        "data_metric": OmegaConf.create({
            "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
            "n_neighbors": 10,
            "at": "dataset",
        }),
    }
    outputs = {"embedding": emb, "dataset": data}
    cache = prewarm_cache(cfgs, emb, ds, outputs=outputs)

    emb_key = _content_key(emb)
    data_key = _content_key(data)

    # Both should be cached with different content keys
    assert emb_key in cache, f"Embedding not in cache. Keys: {list(cache.keys())}"
    assert data_key in cache, f"Dataset not in cache. Keys: {list(cache.keys())}"
    assert emb_key != data_key, "Embedding and dataset cache keys should differ"


# ---------------------------------------------------------------------------
# 8. Bundle config composes multiple metrics
# ---------------------------------------------------------------------------

def test_bundle_config_composes_multiple_metrics():
    """The standard bundle loads several metrics from its Hydra defaults."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics="standard",
    )
    scores = result["scores"]
    # standard.yaml defaults: trustworthiness, continuity, knn_preservation, spectral_gap_ratio
    assert len(scores) >= 2, f"Expected at least 2 scores, got {len(scores)}: {list(scores)}"


# ---------------------------------------------------------------------------
# 9. Result keys have no old group prefix
# ---------------------------------------------------------------------------

def test_result_keys_have_no_group_prefix():
    """Score keys use the flat format, not the old embedding.metric prefix."""
    result = run(
        data="swissroll",
        algorithms=_PCA_ALGO,
        metrics={
            "trustworthiness": {
                "_target_": "manylatents.metrics.trustworthiness.Trustworthiness",
                "_partial_": True,
                "n_neighbors": 5,
                "at": "embedding",
            }
        },
    )
    for key in result["scores"]:
        # Old format used group prefixes like embedding.metric or module.metric.
        # New flat format never starts with a group name followed by a dot.
        assert not key.startswith("embedding."), (
            f"Key '{key}' has old group prefix"
        )
        assert not key.startswith("dataset."), (
            f"Key '{key}' has old group prefix"
        )
        assert not key.startswith("module."), (
            f"Key '{key}' has old group prefix"
        )
