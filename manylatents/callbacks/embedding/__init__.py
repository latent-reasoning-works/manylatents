"""Embedding-level callbacks for post-processing latent outputs."""
from manylatents.callbacks.embedding.base import EmbeddingCallback, EmbeddingOutputs
from manylatents.callbacks.embedding.save_embeddings import SaveEmbeddings
from manylatents.callbacks.embedding.save_trajectory import SaveTrajectory, load_trajectory
from manylatents.callbacks.embedding.loadings_analysis import LoadingsAnalysisCallback

__all__ = [
    "EmbeddingCallback",
    "EmbeddingOutputs",
    "SaveEmbeddings",
    "SaveTrajectory",
    "LoadingsAnalysisCallback",
    "load_trajectory",
]
