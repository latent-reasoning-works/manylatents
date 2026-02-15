from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric


@register_metric(
    aliases=["anisotropy"],
    default_params={},
    description="Anisotropy of embedding space",
)
def Anisotropy(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    cache: Optional[dict] = None,
) -> float:
    """
    Compute anisotropy as the ratio of the first singular value 
    to the sum of all singular values of embeddings.
    Values closer to 1 indicate more anisotropy.
    """
    # Center embeddings
    centered_embeddings = embeddings - np.mean(embeddings, axis=0)

    # Compute singular values
    _, s, _ = np.linalg.svd(centered_embeddings, full_matrices=False)

    anisotropy_ratio = s[0] / np.sum(s)
    return float(anisotropy_ratio)
