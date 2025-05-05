from typing import Optional

import numpy as np
import torch

from src.algorithms.dimensionality_reduction import DimensionalityReductionModule


def Anisotropy(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[DimensionalityReductionModule] = None,
) -> float:
    """
    Compute anisotropy as the ratio of the first singular value 
    to the sum of all singular values of embeddings.
    Values closer to 1 indicate more anisotropy.
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()

    # Center embeddings
    centered_embeddings = embeddings - np.mean(embeddings, axis=0)

    # Compute singular values
    _, s, _ = np.linalg.svd(centered_embeddings, full_matrices=False)

    anisotropy_ratio = s[0] / np.sum(s)
    return float(anisotropy_ratio)
