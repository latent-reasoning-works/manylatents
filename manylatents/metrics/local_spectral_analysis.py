import logging
from typing import Any, Optional, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


# Local spectral analysis: SVD of each point's kNN neighborhood.
# The eigenvalue spectrum is the core object. Summary statistics are modes:
#   - participation_ratio (scalar): (sum lambda)^2 / sum(lambda^2) — eigenvalue-based LID
#   - full: spectrum + effective rank + anisotropy
#
# This is the eigenvalue-based LID estimate. It answers "how many dimensions
# carry signal" via local PCA. Complementary to the distance-based LID
# (LocalIntrinsicDimensionality), which answers "how many dimensions exist"
# via kNN distance ratios. When these two disagree, the manifold has
# anisotropic structure — know this when creating divergence plots and
# reliability maps.
@register_metric(
    aliases=["local_spectral_analysis", "participation_ratio", "pr"],
    default_params={"return_per_sample": False, "output_mode": "scalar"},
    description="Local spectral analysis of kNN neighborhoods (participation ratio mode)",
)
@register_metric(
    aliases=["eigenvalue_effective_rank", "effective_rank", "local_effective_rank"],
    default_params={"output_mode": "full", "n_neighbors": 20},
    description="Per-point eigenvalue effective rank from local PCA on kNN neighborhood",
)
def LocalSpectralAnalysis(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    return_per_sample: bool = False,
    output_mode: str = "scalar",
    cache: Optional[dict] = None,
) -> Union[float, np.ndarray, dict[str, Any]]:
    """
    Local spectral analysis of kNN neighborhoods.

    Core operation: for each point, compute SVD of centered kNN neighborhood
    to get the local eigenvalue spectrum. Summary statistics are selected
    via output_mode.

    output_mode="scalar": participation ratio (sum lambda)^2 / sum(lambda^2).
        Returns float (mean) or per-sample array depending on return_per_sample.
    output_mode="full": returns dict with effective_rank (= participation ratio),
        top_eigenvalue_ratio (anisotropy), and full eigenvalue spectrum per point.

    Args:
        embeddings:   (n_samples, n_features) array (always numpy).
        dataset:      (unused) kept for Protocol compatibility.
        module:       (unused) kept for Protocol compatibility.
        n_neighbors:  how many neighbors (k) to use.
        return_per_sample:
                      if True and output_mode="scalar", returns array of shape
                      (n_samples,) with each sample's PR; else returns the average PR.
        output_mode:  "scalar" (default) or "full".
        cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
        float, np.ndarray, or dict depending on output_mode and return_per_sample.
    """
    n_samples = embeddings.shape[0]

    _, all_idx = compute_knn(embeddings, k=n_neighbors, include_self=True, cache=cache)

    k = all_idx.shape[1] - 1
    chunk_size = max(1, min(10_000, int(2e9 / (k * embeddings.shape[1] * 4))))

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[all_idx[start:end, 1:]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

    s = np.concatenate(sv_chunks, axis=0)

    # Eigenvalues of covariance are proportional to s^2
    s2 = s * s
    total = s2.sum(axis=1)
    sum_sq = (s2 * s2).sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    safe_sum_sq = np.where(sum_sq > 0, sum_sq, 1.0)
    pr_arr = np.where(total > 0, safe_total * safe_total / safe_sum_sq, 0.0)

    if output_mode == "full":
        top_eigenvalue_ratio = np.where(
            total > 0, s2[:, 0] / safe_total, 0.0
        )
        result = {
            "mean_effective_rank": float(np.mean(pr_arr)),
            "std_effective_rank": float(np.std(pr_arr)),
            "mean_top_eigenvalue_ratio": float(np.mean(top_eigenvalue_ratio)),
            "effective_rank": pr_arr,
            "top_eigenvalue_ratio": top_eigenvalue_ratio,
            "eigenvalues": s2,
        }
        logger.info(
            f"LocalSpectralAnalysis(full): mean_rank={result['mean_effective_rank']:.3f}, "
            f"mean_anisotropy={result['mean_top_eigenvalue_ratio']:.3f}"
        )
        return result

    if return_per_sample:
        logger.info(f"LocalSpectralAnalysis: per-sample PR, mean={pr_arr.mean():.3f}")
        return pr_arr

    avg_pr = float(np.nanmean(pr_arr))
    logger.info(f"LocalSpectralAnalysis: average PR = {avg_pr:.3f}")
    return avg_pr


# Backwards compatibility
ParticipationRatio = LocalSpectralAnalysis
