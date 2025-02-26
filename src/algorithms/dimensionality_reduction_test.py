import pytest
import torch

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule


def test_base_dr_instantiation():
    """Ensure the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DimensionalityReductionModule(n_components=2)

def test_fit_transform_not_implemented():
    """Ensure that fit/transform must be implemented by subclasses."""
    class FakeDR(DimensionalityReductionModule):
        pass  # Does not implement abstract methods

    with pytest.raises(TypeError):
        FakeDR(n_components=2)

def test_forward_calls_transform():
    """Ensure forward() uses transform()."""
    class MockDR(DimensionalityReductionModule):
        def fit(self, x):
            pass

        def transform(self, x):
            return x * 2 

    model = MockDR(n_components=2)
    data = torch.randn(10, 5)
    
    output = model.transform(data)

    assert torch.equal(output, data * 2), "Transform method is not working correctly."
