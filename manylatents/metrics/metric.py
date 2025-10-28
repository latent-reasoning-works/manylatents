from typing import Optional, Protocol, Union, Any

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule


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

    Optionally, a dataset and a dimensionality reduction module can be passed to enable
    affinity and kernel matrix specific metrics.
    """
    def __call__(self,
                 embeddings: np.ndarray,
                 dataset: Optional[object] = None,
                 module: Optional[LatentModule] = None
            ) -> Union[float, tuple[float, np.ndarray], dict[str, Any]]:
        ...