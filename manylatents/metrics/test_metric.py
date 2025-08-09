from typing import Optional

import numpy as np


def TestMetric(embeddings: np.ndarray, 
               dataset: Optional[object] = None, 
               module: Optional[object] = None
            ) -> float:
        """
        A test-specific metric that always returns 0.0.
        """
        return 0.0