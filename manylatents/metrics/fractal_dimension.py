import logging

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)

def FractalDimension(dataset, embeddings: np.ndarray, n_box_sizes: int = 10) -> float:
    """
    Estimate the fractal (box-counting) dimension of the embedding.
    
    Parameters:
      - dataset: Provided for protocol compliance.
      - embeddings: A numpy array or torch tensor of shape (n_samples, n_features).
      - n_box_sizes: Number of scales to use in the box-counting method.
      
    Returns:
      - Estimated fractal dimension as a float.
    """
    # Compute minimum and maximum along each feature axis.
    try:
        mins = np.min(embeddings, axis=0)
        maxs = np.max(embeddings, axis=0)
    except Exception as e:
        logger.error("FractalDimension: Error computing min/max of embeddings", exc_info=True)
        raise e

    sizes = maxs - mins
    # Define logarithmically spaced box sizes.
    try:
        epsilons = np.logspace(np.log10(np.min(sizes) / 10), np.log10(np.max(sizes)), num=n_box_sizes)
    except Exception as e:
        logger.error("FractalDimension: Error computing logspace for epsilons", exc_info=True)
        raise e

    counts = []
    for eps in epsilons:
        # Create bins for each dimension.
        bins = [np.arange(mins[i], maxs[i] + eps, eps) for i in range(embeddings.shape[1])]
        hist, _ = np.histogramdd(embeddings, bins=bins)
        counts.append(np.sum(hist > 0))
    try:
        log_eps = np.log(epsilons)
        log_counts = np.log(counts)
        slope, _, _, _, _ = linregress(log_eps, log_counts)
    except Exception as e:
        logger.error("FractalDimension: Error during linear regression", exc_info=True)
        raise e

    logger.info(f"FractalDimension: Computed slope: {slope}")
    return -slope
