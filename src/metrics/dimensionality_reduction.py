import numpy as np
from sklearn.manifold import trustworthiness

from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, **kwargs):
        """
        Compute the metric using provided keyword arguments.
        Returns:
            The computed metric value.
        """
        pass

class Trustworthiness(BaseMetric):
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        """
        Initialize the trustworthiness metric.

        Args:
            n_neighbors (int): Number of neighbors to consider.
            metric (str): The distance metric to use.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric

    def compute(self, original: np.ndarray, embedded: np.ndarray, **kwargs) -> float:
        """
        Compute the trustworthiness score.

        Args:
            original (np.ndarray): The original high-dimensional data.
            embedded (np.ndarray): The low-dimensional embedding.

        Returns:
            float: The trustworthiness score.
        """
        return trustworthiness(original, embedded, n_neighbors=self.n_neighbors, metric=self.metric)
