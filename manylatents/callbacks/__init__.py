"""Callbacks for manylatents experiment pipeline."""
from manylatents.callbacks.base import BaseCallback
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.callbacks.embedding.save_embeddings import SaveEmbeddings
from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback

__all__ = [
    "BaseCallback",
    "EmbeddingCallback",
    "SaveEmbeddings",
    "LoadingsAnalysisCallback",
]
