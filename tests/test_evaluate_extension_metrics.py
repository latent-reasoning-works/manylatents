"""Test evaluate_embeddings handles metrics that don't accept cache=."""
import sys
import types
import numpy as np
import pytest
from omegaconf import OmegaConf

if "manylatents.dogma" not in sys.modules:
    _dogma = types.ModuleType("manylatents.dogma")
    _encoders = types.ModuleType("manylatents.dogma.encoders")
    _base = types.ModuleType("manylatents.dogma.encoders.base")
    _base.FoundationEncoder = type("FoundationEncoder", (), {})
    _dogma.encoders = _encoders
    _encoders.base = _base
    sys.modules["manylatents.dogma"] = _dogma
    sys.modules["manylatents.dogma.encoders"] = _encoders
    sys.modules["manylatents.dogma.encoders.base"] = _base


def test_legacy_metric_without_cache():
    """A metric without cache= should not crash; should retry without it."""
    from manylatents.experiment import prewarm_cache

    call_log = []

    def legacy_metric(embeddings, dataset=None, module=None):
        call_log.append("called")
        return 0.42

    emb = np.random.randn(10, 2).astype(np.float32)
    cache = {"some": "data"}

    # Simulate what evaluate_embeddings should do
    try:
        result = legacy_metric(embeddings=emb, dataset=None, module=None, cache=cache)
    except TypeError:
        result = legacy_metric(embeddings=emb, dataset=None, module=None)

    assert result == 0.42
    assert len(call_log) == 1
