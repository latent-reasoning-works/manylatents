import logging

import numpy as np
from ripser import ripser

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["beta_0", "betti_0"],
    default_params={"homology_dim": 0},
    description="Count of connected components (H0 Betti number)",
)
@register_metric(
    aliases=["beta_1", "betti_1"],
    default_params={"homology_dim": 1},
    description="Count of loops/cycles (H1 Betti number)",
)
def PersistentHomology(embeddings: np.ndarray, 
                       dataset=None, 
                       module=None,
                       homology_dim: int = 1, 
                       persistence_threshold: float = 0.1) -> float:
    """
    Compute a persistent homology metric for the embedding.

    This function uses ripser to compute the persistent diagram and then counts
    the number of features in the specified homology dimension whose persistence
    (death - birth) exceeds a given threshold.

    Parameters:
      - embeddings: A numpy array (or torch tensor) representing the low-dimensional embeddings.
      - dataset: Provided for protocol compliance (unused).
      - module: Provided for protocol compliance (unused).
      - homology_dim: Homology dimension to analyze (e.g., 0 for connected components, 1 for loops).
      - persistence_threshold: Minimum persistence for a feature to be counted.

    Returns:
      - Number of persistent features (float).
    """
    diagrams = ripser(embeddings)['dgms']
    features = diagrams[homology_dim]
    persistence = features[:, 1] - features[:, 0]
    count = np.sum(persistence > persistence_threshold)
    logger.info(f"PersistentHomology: Found {count} features with persistence > {persistence_threshold}")
    return float(count)
