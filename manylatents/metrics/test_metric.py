from typing import Optional, Union, List

import numpy as np


def TestMetric(embeddings: np.ndarray,
               dataset: Optional[object] = None,
               module: Optional[object] = None,
               k: Union[int, List[int]] = 25
            ) -> tuple[float, np.ndarray]:
        """
        A test-specific metric that always returns 0.0 for both scalar and per-sample scores.

        This metric is designed for:
        - Fast smoke testing of the metrics pipeline
        - Validating wandb integration (scalar + table logging)
        - Testing all metric groups (dataset/embedding/module)

        Args:
            embeddings: The embedding array
            dataset: Optional dataset object
            module: Optional module object
            k: Number of neighbors (can be int or list for sweeping)

        Returns:
            Tuple of (scalar, per_sample):
                - scalar: 0.0 (aggregate metric)
                - per_sample: Array of zeros (one per sample)
        """
        n_samples = len(embeddings)
        per_sample_scores = np.zeros(n_samples)
        return (0.0, per_sample_scores)