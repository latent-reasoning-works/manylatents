from typing import Dict, Optional, Protocol, Tuple, Union, Any

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule


# Type alias for precomputed kNN cache
KNNCache = Optional[Tuple[np.ndarray, np.ndarray]]
"""
Precomputed kNN cache: (distances, indices) arrays.

Both arrays have shape (n_samples, max_k+1) where index 0 is the point itself.
Metrics should slice to their required k: distances[:, 1:k+1], indices[:, 1:k+1]

Passed by evaluate_embeddings() when multiple metrics share kNN computation.
"""

# Type alias for precomputed SVD cache (shared local neighborhood decomposition)
SVDCache = Optional[Dict[int, np.ndarray]]
"""
Precomputed SVD cache: {k: singular_values} mapping.

Each value is a singular values array of shape (n_samples, min(k, d)) from the SVD
of centered kNN neighborhoods. Keyed by k (number of neighbors, excluding self).

Used by ParticipationRatio and TangentSpaceApproximation to avoid redundant SVD
computations on identical neighborhoods. Computed once by compute_svd_cache() in
experiment.py and passed to metrics that accept a _svd_cache parameter.

If None, metrics compute SVD inline (backward compatible).
"""


class Metric(Protocol):
    """Protocol for metrics that evaluate embeddings.

    A metric is a callable that takes embeddings as input and returns one of:

    1. float: Simple scalar metric (e.g., 0.95)
       - Use for: Basic summary statistics

    2. tuple[float, np.ndarray]: Scalar + per-sample values
       - Use for: Metrics with per-sample breakdowns (enables wandb table logging)
       - Example: (0.85, np.array([0.9, 0.8, 0.75, ...]))

    3. dict[str, Any]: Structured output
       - Use for: Complex metrics (graphs, multiple values, etc.)
       - Example: {'nodes': [...], 'edges': [...]}

    Standard parameters:
        embeddings: Low-dimensional embedding array (n_samples, n_dims)
        dataset: Dataset object with .data attribute for high-dimensional data
        module: Fitted LatentModule instance (for accessing affinity matrices, etc.)

    Optional cache parameters:
        _knn_cache: Precomputed (distances, indices) from shared kNN computation.
            Shape: (n_samples, max_k+1) including self at index 0.
            Metrics that use kNN should accept this and slice to their k value.
            If None, metric computes its own kNN.

        _svd_cache: Precomputed {k: singular_values} from shared SVD computation.
            Singular values shape: (n_samples, min(k, d)).
            Metrics that do local PCA/SVD on kNN neighborhoods should accept this.
            If None, metric computes SVD inline.
    """
    def __call__(
        self,
        embeddings: np.ndarray,
        dataset: Optional[object] = None,
        module: Optional[LatentModule] = None,
        _knn_cache: KNNCache = None,
        _svd_cache: SVDCache = None,
    ) -> Union[float, Tuple[float, np.ndarray], dict[str, Any]]:
        ...