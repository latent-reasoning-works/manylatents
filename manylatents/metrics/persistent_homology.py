import logging
import random
from typing import Optional

import numpy as np
import torch
from ripser import ripser
from scipy.spatial.distance import pdist

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
                       threshold_mode: str = "relative",
                       max_N: Optional[int] = 2000,
                       random_seed: int = 0,
                       output_mode: str = "count",
                       cache: Optional[dict] = None):
    """
    Compute a persistent homology metric for the embedding.

    Parameters:
      - embeddings: Embedding array (or torch tensor).
      - homology_dim: 0 for connected components, 1 for loops.
      - persistence_threshold: In "relative" mode (the default), a fraction of the point
            cloud's DIAMETER. In "absolute" mode, a cutoff in the embedding's own coordinate
            units — which means the answer depends on how the embedding is scaled, so prefer
            relative unless you are deliberately sweeping an absolute threshold.
      - threshold_mode: "relative" (default) uses persistence_threshold * diameter;
            "absolute" uses persistence_threshold directly. Relative is scale-free, so the
            same topology scores the same whether the coordinates span 1e-2 (PHATE) or 1e1
            (PCA of raw features) — which is what makes the number comparable across methods.
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

    finite_persistence = persistence[np.isfinite(persistence)]
    max_pers = float(finite_persistence.max()) if len(finite_persistence) > 0 else 0.0

    if threshold_mode == "relative":
        # A fraction of the cloud's DIAMETER. The cutoff used to be either an absolute
        # constant (0.1 in the units of whatever coordinates arrived) or a fraction of the max
        # finite persistence. Both are wrong for different reasons.
        #
        # Absolute: Rips persistence carries the embedding's units, so rescaling one fixed
        # 3-blob cloud swept betti_0 across 197 / 139 / 2 / 0 / 0 at scales 10 / 1 / 0.1 /
        # 0.01 / 0.001 — five answers for one topology. PHATE lives at ~1e-2, so it returned
        # 0/0 for everything, which is every embedding geomancer ships.
        #
        # Fraction of max persistence: scale-invariant but self-referential — the largest gap
        # is the very feature being measured, giving a constant 85/26 on that same cloud.
        #
        # Diameter is the honest denominator: scale-free and independent of the diagram.
        diameter = float(pdist(X).max()) if len(X) > 1 else 0.0
        cutoff = persistence_threshold * diameter
    else:
        cutoff = persistence_threshold

    # Count over the FULL array, not just the finite bars. The essential class (the
    # whole-cloud component, persistence = inf) was being dropped, which made betti_0
    # systematically one low and let it reach 0 — topologically impossible for a non-empty
    # cloud, and the reason a single point scored 0 instead of 1.
    count = float(np.sum(persistence > cutoff))
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
