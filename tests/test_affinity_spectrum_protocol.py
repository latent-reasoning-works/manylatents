"""Test AffinitySpectrum conforms to Metric protocol."""
import inspect
import numpy as np
import pytest


def test_affinity_spectrum_embeddings_first():
    """First parameter must be 'embeddings'."""
    from manylatents.metrics.affinity_spectrum import AffinitySpectrum
    params = list(inspect.signature(AffinitySpectrum).parameters.keys())
    assert params[0] == "embeddings"


def test_affinity_spectrum_optional_dataset_module():
    """dataset and module should default to None."""
    from manylatents.metrics.affinity_spectrum import AffinitySpectrum
    sig = inspect.signature(AffinitySpectrum)
    assert sig.parameters["dataset"].default is None
    assert sig.parameters["module"].default is None


def test_affinity_spectrum_positional_call():
    """Calling with just embeddings must not raise."""
    from manylatents.metrics.affinity_spectrum import AffinitySpectrum
    result = AffinitySpectrum(np.random.randn(10, 2).astype(np.float32))
    assert isinstance(result, np.ndarray)
