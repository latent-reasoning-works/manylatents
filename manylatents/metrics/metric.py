from typing import Optional, Protocol, Tuple, Union, Any

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

    Optional kNN cache parameter:
        _knn_cache: Precomputed (distances, indices) from shared kNN computation.
            Shape: (n_samples, max_k+1) including self at index 0.
            Metrics that use kNN should accept this and slice to their k value.
            If None, metric computes its own kNN.
    """
    def __call__(
        self,
        embeddings: np.ndarray,
        dataset: Optional[object] = None,
        module: Optional[LatentModule] = None,
        _knn_cache: KNNCache = None,
    ) -> Union[float, Tuple[float, np.ndarray], dict[str, Any]]:
        ...