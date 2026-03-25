"""Callbacks for manylatents experiment pipeline."""
from manylatents.callbacks.callback import BaseCallback
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.callbacks.embedding.save_outputs import SaveOutputs, SaveEmbeddings
from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback

__all__ = [
    "BaseCallback",
    "EmbeddingCallback",
    "SaveOutputs",
    "SaveEmbeddings",  # backwards-compat alias
    "LoadingsAnalysisCallback",
]
