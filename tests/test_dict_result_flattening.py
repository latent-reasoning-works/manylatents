import numpy as np


def test_dict_metric_result_is_flattened():
    """When a metric returns a dict, _flatten_metric_result should expand it."""
    from manylatents.evaluate import _flatten_metric_result
    mock_result = {
        "mean_score": 0.95,
        "per_point": np.ones(50),
    }
    flat = _flatten_metric_result("embedding.my_metric", mock_result)
    assert "embedding.my_metric.mean_score" in flat
    assert flat["embedding.my_metric.mean_score"] == 0.95
    assert "embedding.my_metric.per_point" in flat
    assert isinstance(flat["embedding.my_metric.per_point"], np.ndarray)


def test_scalar_result_unchanged():
    """Scalar results should pass through unchanged."""
    from manylatents.evaluate import _flatten_metric_result
    flat = _flatten_metric_result("embedding.lid", 3.5)
    assert flat == {"embedding.lid": 3.5}


def test_tuple_result_unchanged():
    """Non-dict results should pass through unchanged."""
    from manylatents.evaluate import _flatten_metric_result
    flat = _flatten_metric_result("embedding.lid", (3.5, "viz"))
    assert flat == {"embedding.lid": (3.5, "viz")}
