from typing import Any, Dict, Optional, Protocol, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule


class Metric(Protocol):
    """Protocol for metrics that evaluate embeddings.

    A metric is a callable that takes embeddings as input and returns one of:

    1. float: Simple scalar metric (e.g., 0.95)
    2. tuple[float, np.ndarray]: Scalar + per-sample values
    3. dict[str, Any]: Structured output

    Standard parameters:
        embeddings: Low-dimensional embedding array (n_samples, n_dims)
        dataset: Dataset object with .data attribute for high-dimensional data
        module: Fitted LatentModule instance (for accessing affinity matrices, etc.)

    Cache parameter:
        cache: Optional dict shared across metrics within one evaluation run.
            Metrics should pass this through to compute_knn() and
            compute_eigenvalues() from manylatents.utils.metrics.
            Do NOT slice the cache directly — call the utility functions,
            which handle cache lookup and population internally.

            If None, utility functions compute from scratch (backward compatible).
    """
    def __call__(
        self,
        embeddings: np.ndarray,
        dataset: Optional[object] = None,
        module: Optional[LatentModule] = None,
        cache: Optional[dict] = None,
    ) -> Union[float, Tuple[float, np.ndarray], dict[str, Any]]:
        ...
