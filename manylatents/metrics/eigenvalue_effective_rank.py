import logging
from typing import Any, Optional

import numpy as np

from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["eigenvalue_effective_rank", "effective_rank", "local_effective_rank"],
    default_params={"k": 20},
    description="Per-point eigenvalue effective rank from local PCA on kNN neighborhood",
)
def EigenvalueEffectiveRank(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[object] = None,
    k: int = 20,
    cache: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Per-point eigenvalue effective rank from local PCA on kNN neighborhood.

    For each point, takes k nearest neighbors, computes SVD of centered
    neighborhood, and derives:
    - Effective rank via participation ratio: (sum lambda)^2 / sum(lambda^2)
    - Top eigenvalue ratio: lambda_1 / sum(lambda) (anisotropy indicator)
    - Full local eigenvalue spectrum

    Args:
        embeddings: (n_samples, n_features) array.
        dataset: Unused, kept for protocol compatibility.
        module: Unused, kept for protocol compatibility.
        k: Number of neighbors for local PCA.
        cache: Optional shared cache dict for kNN reuse.

    Returns:
        Dict with summary scalars and per-point arrays.
    """
    n_samples, d = embeddings.shape
    _, all_idx = compute_knn(embeddings, k=k, include_self=True, cache=cache)

    # Batched SVD (same pattern as ParticipationRatio)
    kk = all_idx.shape[1] - 1  # actual k (may be clamped)
    chunk_size = max(1, min(10_000, int(2e9 / (kk * d * 4))))

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[all_idx[start:end, 1:]]  # (chunk, k, d)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

    s = np.concatenate(sv_chunks, axis=0)  # (n_samples, min(k, d))

    # Eigenvalues of local covariance = squared singular values
    eigenvalues = s * s  # (n_samples, min(k, d))

    # Participation ratio: (sum lambda)^2 / sum(lambda^2)
    total = eigenvalues.sum(axis=1)
    sum_sq = (eigenvalues ** 2).sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    safe_sum_sq = np.where(sum_sq > 0, sum_sq, 1.0)
    effective_rank = np.where(total > 0, safe_total ** 2 / safe_sum_sq, 0.0)

    # Anisotropy: lambda_1 / sum(lambda)
    top_eigenvalue_ratio = np.where(
        total > 0, eigenvalues[:, 0] / safe_total, 0.0
    )

    result = {
        "mean_effective_rank": float(np.mean(effective_rank)),
        "std_effective_rank": float(np.std(effective_rank)),
        "mean_top_eigenvalue_ratio": float(np.mean(top_eigenvalue_ratio)),
        "effective_rank": effective_rank,
        "top_eigenvalue_ratio": top_eigenvalue_ratio,
        "eigenvalues": eigenvalues,
    }

    logger.info(
        f"EigenvalueEffectiveRank: mean_rank={result['mean_effective_rank']:.3f}, "
        f"mean_anisotropy={result['mean_top_eigenvalue_ratio']:.3f}"
    )
    return result
