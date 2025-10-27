"""
Integration tests for manylatents-omics extension package.

These tests verify that the manylatents-omics namespace package works correctly
and can be imported alongside core manylatents.

Run with: pytest manylatents/test_omics_integration.py

Note: Requires manylatents-omics to be installed:
    uv pip install git+https://github.com/latent-reasoning-works/manylatents-omics.git
or:
    pip install git+https://github.com/latent-reasoning-works/manylatents-omics.git
"""

import pytest


def test_omics_package_installed():
    """Test that manylatents-omics package is installed."""
    try:
        import manylatents.omics
    except ImportError as e:
        pytest.skip(f"manylatents-omics not installed: {e}")
    
    # If we get here, it's installed
    assert manylatents.omics.__version__ == "0.1.0"


def test_omics_namespace_package():
    """Test that omics works as a namespace package extending manylatents."""
    try:
        import manylatents.omics
    except ImportError:
        pytest.skip("manylatents-omics not installed")
    
    # Verify it has the expected modules
    assert hasattr(manylatents.omics, 'data')
    assert hasattr(manylatents.omics, 'metrics')
    assert hasattr(manylatents.omics, 'callbacks')
    assert hasattr(manylatents.omics, 'utils')


def test_omics_data_imports():
    """Test that genetics datasets can be imported."""
    try:
        from manylatents.omics.data import (
            PlinkDataset,
            HGDPDataset,
            HGDPDataModule,
            AOUDataset,
            AOUDataModule,
            UKBBDataset,
            UKBBDataModule,
        )
    except ImportError:
        pytest.skip("manylatents-omics not installed")
    
    # Verify classes exist
    assert PlinkDataset is not None
    assert HGDPDataset is not None
    assert HGDPDataModule is not None
    assert AOUDataset is not None
    assert AOUDataModule is not None
    assert UKBBDataset is not None
    assert UKBBDataModule is not None


def test_omics_metrics_imports():
    """Test that genetics metrics can be imported."""
    try:
        from manylatents.omics.metrics import (
            GeographicPreservation,
            AdmixturePreservation,
            AdmixturePreservationK,
            sample_id,
        )
    except ImportError:
        pytest.skip("manylatents-omics not installed")
    
    # Verify functions exist
    assert callable(GeographicPreservation)
    assert callable(AdmixturePreservation)
    assert callable(AdmixturePreservationK)
    assert callable(sample_id)


def test_core_and_omics_coexist():
    """Test that core manylatents and omics can be imported together."""
    try:
        import manylatents.omics  # noqa: F401
    except ImportError:
        pytest.skip("manylatents-omics not installed")
    
    # Import from core
    from manylatents.data import SwissRoll
    from manylatents.metrics import trustworthiness
    
    # Import from omics
    from manylatents.omics.data import PlinkDataset
    from manylatents.omics.metrics import GeographicPreservation
    
    # Verify they're different modules
    assert SwissRoll is not None
    assert PlinkDataset is not None
    assert trustworthiness is not None
    assert GeographicPreservation is not None


def test_omics_precomputed_mixin():
    """Test that PrecomputedMixin is available in omics."""
    try:
        from manylatents.omics.data import PrecomputedMixin
    except ImportError:
        pytest.skip("manylatents-omics not installed")
    
    # Verify it has the expected methods
    assert hasattr(PrecomputedMixin, 'load_precomputed')
    assert hasattr(PrecomputedMixin, 'get_data')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
