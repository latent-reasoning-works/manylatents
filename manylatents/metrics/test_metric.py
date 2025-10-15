from typing import Optional, Union, List, Tuple

import numpy as np


def TestMetric(embeddings: np.ndarray,
               dataset: Optional[object] = None,
               module: Optional[object] = None,
               k: Union[int, List[int]] = 25
            ) -> Tuple[float, np.ndarray]:
        """
        A test-specific metric that always returns 0.0 for both scalar and per-sample scores.
        Returns a tuple of (scalar, per_sample_array) to enable wandb table logging.
        Now supports k parameter for sweeping.

        Args:
            embeddings: The embedding array
            dataset: Optional dataset object
            module: Optional module object
            k: Number of neighbors (can be int or list for sweeping)

        Returns:
            Tuple of (scalar_score, per_sample_scores):
                - scalar_score: Aggregate metric (always 0.0)
                - per_sample_scores: Array of per-sample metrics (all zeros)
        """
        n_samples = len(embeddings)
        per_sample_scores = np.zeros(n_samples)
        return (0.0, per_sample_scores)