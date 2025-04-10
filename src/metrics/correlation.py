import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

def PearsonCorrelation(dataset, embeddings: np.ndarray,
                       return_per_sample: bool = False, 
                       num_dists: int = 100,
                       random_state: int = 42) -> float:
    """
    Compute the Pearson correlation between pairwise distances in the 
    original high-dimensional data and the low-dimensional embeddings.

    If return_per_sample is False, returns a single global Pearson correlation (float).
    If return_per_sample is True, returns a numpy array of Pearson correlations 
    for each individual (i.e., correlation between the distances from each individual 
    to all others in the original data and in the embedding).

    Args:
        dataset: Object with an attribute `data` containing the original high-dimensional data.
        embeddings: Numpy array of low-dimensional embeddings (n_samples x dims).
        return_per_sample: If True, return per-individual correlations; otherwise, return a single value.
        num_dists: When computing a global correlation, subsample at most this many pairwise distances.

    Returns:
        A float if return_per_sample is False, or a numpy array of shape (n_samples,) if True.
    """
    if dataset.data.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: dataset has {dataset.data.shape[0]} samples but embeddings has {embeddings.shape[0]}"
        )

    if not return_per_sample:
        # Compute the global Pearson correlation using a random subset of distances.
        logger.info(f"Starting global Pearson correlation computation with {num_dists} distances.")
        orig_dists = pdist(dataset.data)
        emb_dists = pdist(embeddings)
        
        n = len(emb_dists)
        subset_size = min(num_dists, n)
        indices = np.random.choice(n, subset_size, replace=False)
        orig_sample = orig_dists[indices]
        emb_sample = emb_dists[indices]
        
        corr, _ = pearsonr(orig_sample, emb_sample)
        logger.info("Finished global Pearson correlation computation.")
        return corr
    
    else:
        # Compute per-individual Pearson correlations.
        logger.info("Starting per-individual Pearson correlation computation.")
        orig_dists = squareform(pdist(dataset.data))
        emb_dists = squareform(pdist(embeddings))
        
        n = orig_dists.shape[0]
        correlations = np.empty(n)
        for i in range(n):
            # Remove self-distance.
            orig_row = np.delete(orig_dists[i, :], i)
            emb_row = np.delete(emb_dists[i, :], i)
            corr, _ = pearsonr(orig_row, emb_row)
            correlations[i] = corr
        
        logger.info("Finished per-individual Pearson correlation computation.")
        return correlations
