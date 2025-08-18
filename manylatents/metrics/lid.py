import logging

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def LocalIntrinsicDimensionality(embeddings: np.ndarray, 
                                 dataset=None, 
                                 module=None,
                                 k: int = 20) -> float:
    """
    Compute the mean Local Intrinsic Dimensionality (LID) for the embedding.

    Parameters:
      - embeddings: A numpy array (or torch tensor) representing the low-dimensional embeddings.
      - dataset: Provided for protocol compliance (unused).
      - module: Provided for protocol compliance (unused).
      - k: The number of nearest neighbors to consider.

    Returns:
      - A float representing the mean LID.
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
        logger.info(f"LocalIntrinsicDimensionality: Converted embeddings to numpy with shape {embeddings.shape}")
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = distances[:, 1:]  # Exclude the self-distance.
    r_k = distances[:, -1]
    lid_values = -k / np.sum(np.log(distances / r_k[:, None] + 1e-10), axis=1)
    mean_lid = float(np.mean(lid_values))
    logger.info(f"LocalIntrinsicDimensionality: Computed mean LID = {mean_lid}")
    return mean_lid
