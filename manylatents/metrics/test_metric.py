from typing import Optional, Union, List

import numpy as np


def TestMetric(embeddings: np.ndarray, 
               dataset: Optional[object] = None, 
               module: Optional[object] = None,
               k: Union[int, List[int]] = 25
            ) -> float:
        """
        A test-specific metric that always returns 0.0.
        Now supports k parameter for sweeping.
        
        Args:
            embeddings: The embedding array
            dataset: Optional dataset object
            module: Optional module object
            k: Number of neighbors (can be int or list for sweeping)
        """
        return 0.0