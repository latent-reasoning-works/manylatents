"""Embedding-level callbacks for post-processing latent outputs."""
from manylatents.callbacks.embedding.base import EmbeddingCallback, EmbeddingOutputs
from manylatents.callbacks.embedding.save_embeddings import SaveEmbeddings
from manylatents.callbacks.embedding.linear_probe import LinearProbeCallback

__all__ = [
    "EmbeddingCallback",
    "EmbeddingOutputs",
    "SaveEmbeddings",
    "LinearProbeCallback",
]
