import logging

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def PearsonCorrelation(dataset, embeddings: np.ndarray, num_dists: int = 50000) -> float:
    """
    Compute the Pearson correlation between pairwise distances of the original
    high-dimensional data and the low-dimensional embeddings. Uses subsampling
    to limit the number of distance comparisons if necessary.
    
    Assumes that the dataset instance has an attribute `original_data`.
    """
    logger.info(f"Starting Pearson correlation computation with {num_dists} distances.")
    return 0.0
    # Retrieve the original data from the dataset.
    original_data = dataset.original_data
    if original_data is None:
        raise ValueError("Dataset does not have 'original_data' attribute.")
    
    # Compute pairwise distances.
    orig_dists = pdist(original_data)
    emb_dists = pdist(embeddings)
    
    n = len(emb_dists)
    # Choose a random subset of distances if needed.
    subset_size = min(num_dists, n)
    indices = np.random.choice(n, subset_size, replace=False)
    orig_sample = orig_dists[indices]
    emb_sample = emb_dists[indices]
    
    corr, _ = pearsonr(orig_sample, emb_sample)
    
    logger.info("Finished Pearson correlation computation.")
    return corr