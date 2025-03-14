## typing purposes only
from typing import Protocol

import numpy as np


class Metric(Protocol):
    def __call__(self, dataset, embeddings: np.ndarray) -> float: ...