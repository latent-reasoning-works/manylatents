import pytest
import torch
from src.algorithms.dimensionality_reduction import DimensionalityReduction

def test_base_dr_instantiation():
    """Ensure the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DimensionalityReduction(n_components=2)

def test_fit_transform_not_implemented():
    """Ensure that fit/transform must be implemented by subclasses."""
    class FakeDR(DimensionalityReduction):
        pass  # Does not implement abstract methods

    with pytest.raises(TypeError):
        FakeDR(n_components=2)

def test_forward_calls_transform():
    """Ensure forward() uses transform()."""
    class MockDR(DimensionalityReduction):
        def fit(self, x):
            pass

        def transform(self, x):
            return x * 2  # Simple transform for testing

    model = MockDR(n_components=2)
    data = torch.randn(10, 5)
    output = model(data)

    assert torch.equal(output, data * 2), "Forward pass does not use transform correctly."
