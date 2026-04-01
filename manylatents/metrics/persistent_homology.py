import logging
import random
from typing import Optional

import numpy as np
import torch
from ripser import ripser

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["beta_0", "betti_0"],
    default_params={"homology_dim": 0, "max_N": 2000, "random_seed": 0},
    description="Count of connected components (H0 Betti number)",
)
@register_metric(
    aliases=["beta_1", "betti_1"],
    default_params={"homology_dim": 1, "max_N": 2000, "random_seed": 0},
    description="Count of loops/cycles (H1 Betti number)",
)
def PersistentHomology(embeddings: np.ndarray,
                       dataset=None,
                       module=None,
                       homology_dim: int = 1,
                       persistence_threshold: float = 0.1,
                       threshold_mode: str = "absolute",
                       max_N: Optional[int] = 2000,
                       random_seed: int = 0,
                       output_mode: str = "count",
                       cache: Optional[dict] = None):
    """
    Compute a persistent homology metric for the embedding.

    Parameters:
      - embeddings: Embedding array (or torch tensor).
      - homology_dim: 0 for connected components, 1 for loops.
      - persistence_threshold: Minimum persistence for a feature to be counted
            (absolute mode) or fraction of max persistence (relative mode).
      - threshold_mode: "absolute" uses persistence_threshold directly;
            "relative" uses persistence_threshold * max_persistence as cutoff.
            Relative mode is scale-adaptive per embedding, useful when comparing
            fragmentation across methods with different embedding scales.
      - max_N: Subsample to this many points before Rips filtration (O(n^2) memory).
      - random_seed: Seed for reproducible subsampling.
      - output_mode: "count" returns float, "diagrams" returns dict with count + raw diagrams.

    Returns:
      - float (count mode) or dict with "count", "diagrams", "max_persistence",
        and "cutoff" keys (diagrams mode).
    """
    X = embeddings
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    if max_N is not None and len(X) > max_N:
        random.seed(random_seed)
        idx = np.array(random.sample(range(len(X)), k=max_N))
        logger.info(f"PersistentHomology: subsampled {len(X)} → {max_N}")
        X = X[idx]

    diagrams = ripser(X, maxdim=homology_dim)['dgms']
    features = diagrams[homology_dim]
    persistence = features[:, 1] - features[:, 0]

    # Remove infinite bars for threshold computation
    finite_mask = np.isfinite(persistence)
    finite_persistence = persistence[finite_mask]

    max_pers = float(finite_persistence.max()) if len(finite_persistence) > 0 else 0.0

    if threshold_mode == "relative" and max_pers > 0:
        cutoff = persistence_threshold * max_pers
    else:
        cutoff = persistence_threshold

    count = float(np.sum(finite_persistence > cutoff))
    logger.info(
        f"PersistentHomology: {int(count)} features with persistence > {cutoff:.4f} "
        f"({threshold_mode}, max_pers={max_pers:.4f})"
    )

    if output_mode == "diagrams":
        return {
            "count": count,
            "diagrams": diagrams,
            "max_persistence": max_pers,
            "cutoff": cutoff,
        }
    return count
