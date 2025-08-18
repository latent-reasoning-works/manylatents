import logging

import numpy as np
import torch
from ripser import ripser

logger = logging.getLogger(__name__)

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
    # Convert to numpy if necessary
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        logger.info(f"PersistentHomology: Converted embeddings to numpy with shape {embeddings.shape}")
    
    diagrams = ripser(embeddings)['dgms']
    features = diagrams[homology_dim]
    persistence = features[:, 1] - features[:, 0]
    count = np.sum(persistence > persistence_threshold)
    logger.info(f"PersistentHomology: Found {count} features with persistence > {persistence_threshold}")
    return float(count)
