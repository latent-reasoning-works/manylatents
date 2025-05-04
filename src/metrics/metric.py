from typing import Optional, Protocol

import numpy as np

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule


class Metric(Protocol):
    def __call__(self, 
                 embeddings: np.ndarray, 
                 dataset: Optional[object] = None, 
                 module: Optional[DimensionalityReductionModule] = None
            ) -> float:
        ...
    
    """A class that defines a metric for evaluating embeddings, irrespective of their modelling source.
    The metric is defined as a callable that takes embeddings as input and returns a float value.
    Optionally, a dataset and a dimensionality reduction module can be passed to the metric, to enable affinity and kernel matrix specific metrics.
    """