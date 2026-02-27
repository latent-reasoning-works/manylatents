"""Embedding-level callbacks for post-processing latent outputs."""
from manylatents.callbacks.embedding.base import EmbeddingCallback, LatentOutputs, EmbeddingOutputs
from manylatents.callbacks.embedding.save_embeddings import SaveEmbeddings
from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback

__all__ = [
    "EmbeddingCallback",
    "LatentOutputs",
    "EmbeddingOutputs",
    "SaveEmbeddings",
    "LoadingsAnalysisCallback",
]
