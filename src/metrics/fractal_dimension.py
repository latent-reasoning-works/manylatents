import numpy as np
from scipy.stats import linregress


def FractalDimension(dataset, embeddings: np.ndarray, n_box_sizes: int = 10) -> float:
    """
    Estimate the fractal (box-counting) dimension of the embedding.

    Parameters:
      - dataset: Provided for protocol compliance.
      - embeddings: A numpy array of shape (n_samples, n_features).
      - n_box_sizes: Number of scales to use in the box-counting method.

    Returns:
      - Estimated fractal dimension as a float.
    """
    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    sizes = maxs - mins
    # Generate logarithmically spaced box sizes.
    epsilons = np.logspace(np.log10(np.min(sizes) / 10), np.log10(np.max(sizes)), num=n_box_sizes)
    counts = []
    for eps in epsilons:
        # Create bins for each dimension.
        bins = [np.arange(mins[i], maxs[i] + eps, eps) for i in range(embeddings.shape[1])]
        hist, _ = np.histogramdd(embeddings, bins=bins)
        counts.append(np.sum(hist > 0))
    log_eps = np.log(epsilons)
    log_counts = np.log(counts)
    slope, _, _, _, _ = linregress(log_eps, log_counts)
    return -slope  # The fractal dimension is approximated as the negative slope.
