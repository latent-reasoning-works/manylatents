"""Integration tests for GPU metric infrastructure.

Verifies end-to-end pipeline: UMAPModule + spectral metrics + silhouette,
dataset capabilities, backend utilities, and metric importability.
"""
import numpy as np
import pytest
import torch


def test_full_pipeline_cpu():
    """End-to-end: UMAPModule + spectral metrics + silhouette on random data."""
    from manylatents.algorithms.latent.umap import UMAPModule
    from manylatents.metrics.spectral_gap_ratio import SpectralGapRatio
    from manylatents.metrics.spectral_decay_rate import SpectralDecayRate
    from manylatents.metrics.silhouette import SilhouetteScore

    # Need to mock dogma for experiment import
    import sys
    if "manylatents.dogma" not in sys.modules:
        import types
        dogma_mock = types.ModuleType("manylatents.dogma")
        dogma_encoders = types.ModuleType("manylatents.dogma.encoders")
        dogma_base = types.ModuleType("manylatents.dogma.encoders.base")
        dogma_base.FoundationEncoder = type("FoundationEncoder", (), {})
        dogma_mock.encoders = dogma_encoders
        dogma_encoders.base = dogma_base
        sys.modules["manylatents.dogma"] = dogma_mock
        sys.modules["manylatents.dogma.encoders"] = dogma_encoders
        sys.modules["manylatents.dogma.encoders.base"] = dogma_base

    from manylatents.experiment import _compute_eigenvalue_cache

    # Generate simple data
    x = torch.randn(100, 10)

    # Fit UMAP
    m = UMAPModule(n_components=2, random_state=42, n_neighbors=10, n_epochs=50)
    emb = m.fit_transform(x)
    emb_np = emb.numpy()

    # Compute eigenvalue cache
    cache = _compute_eigenvalue_cache(m, top_k_values={None, 20})
    assert (True, None) in cache
    assert (True, 20) in cache

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


def test_dataset_capabilities_integration():
    """get_capabilities works on various fake datasets."""
    from manylatents.data.capabilities import get_capabilities

    class SwissRollLike:
        def get_gt_dists(self): return np.eye(10)
        def get_graph(self): return "graph"
        def get_labels(self): return np.arange(10)

    caps = get_capabilities(SwissRollLike())
    assert caps["gt_dists"] is True
    assert caps["labels"] is True
    assert caps["gt_type"] == "manifold"


def test_backend_utils_integration():
    """Backend utilities work together."""
    from manylatents.utils.backend import (
        check_torchdr_available,
        check_faiss_available,
        resolve_device,
        resolve_backend,
    )

    # These should return booleans
    assert isinstance(check_torchdr_available(), bool)
    assert isinstance(check_faiss_available(), bool)

    # resolve_device should return "cpu" or "cuda"
    assert resolve_device(None) in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"

    # resolve_backend should return None for default
    assert resolve_backend(None) is None
    assert resolve_backend("sklearn") is None


def test_all_new_metrics_importable():
    """All new metrics can be imported from the package."""
    from manylatents.metrics import (
        SpectralGapRatio,
        SpectralDecayRate,
        SilhouetteScore,
        GeodesicDistanceCorrelation,
        DatasetTopologyDescriptor,
        MetricAgreement,
    )
    # Verify they are callable
    assert callable(SpectralGapRatio)
    assert callable(SpectralDecayRate)
    assert callable(SilhouetteScore)
    assert callable(GeodesicDistanceCorrelation)
    assert callable(DatasetTopologyDescriptor)
    assert callable(MetricAgreement)
