"""Cross-modal neighborhood Jaccard overlap for alignment measurement.

Computes k-NN neighborhood overlap between embedding spaces of different
modalities without requiring projection to a common dimension.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.neighbors import NearestNeighbors

from manylatents.metrics.registry import register_metric


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D, squeezing if needed."""
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.squeeze(1)
    return arr


def compute_knn_indices(
    embeddings: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute k-nearest neighbor indices for each sample.

    Args:
        embeddings: Array of shape (N, D).
        k: Number of neighbors to find.
        metric: Distance metric.

    Returns:
        Array of shape (N, k) with neighbor indices.
    """
    embeddings = _ensure_2d(embeddings)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nn.fit(embeddings)
    indices = nn.kneighbors(return_distance=False)[:, 1:]  # Exclude self
    return indices


def cross_modal_jaccard_pairwise(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    k: int = 20,
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute per-sample Jaccard overlap between two embedding spaces.

    Args:
        embeddings_a: Array of shape (N, D1) or (N, 1, D1).
        embeddings_b: Array of shape (N, D2) or (N, 1, D2).
        k: Number of neighbors.
        metric: Distance metric.

    Returns:
        Array of shape (N,) with Jaccard overlap for each sample.
    """
    embeddings_a = _ensure_2d(embeddings_a)
    embeddings_b = _ensure_2d(embeddings_b)

    if embeddings_a.shape[0] != embeddings_b.shape[0]:
        raise ValueError(f"Sample count mismatch: {embeddings_a.shape[0]} vs {embeddings_b.shape[0]}")

    n_samples = embeddings_a.shape[0]
    knn_a = compute_knn_indices(embeddings_a, k=k, metric=metric)
    knn_b = compute_knn_indices(embeddings_b, k=k, metric=metric)

    jaccard_scores = np.zeros(n_samples)
    for i in range(n_samples):
        set_a = set(knn_a[i])
        set_b = set(knn_b[i])
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccard_scores[i] = intersection / union if union > 0 else 0.0

    return jaccard_scores


@register_metric(
    aliases=["neighborhood_jaccard", "cross_modal_overlap"],
    default_params={"k": 20},
    description="Cross-modal k-NN neighborhood Jaccard overlap",
)
def CrossModalJaccard(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    dataset=None,
    module=None,
    k: int = 20,
    metric: str = "euclidean",
    return_per_sample: bool = False,
    return_pairwise: bool = False,
) -> Union[float, np.ndarray, Dict[str, np.ndarray]]:
    """Compute cross-modal k-NN neighborhood overlap.

    Primary alignment measure for comparing embedding spaces across modalities.
    Invariant to ambient dimension - only cares about neighbor identity.

    Args:
        embeddings: Either single array (self-comparison) or dict mapping
            modality names to arrays, e.g., {"esm3": (N, 1536), "evo2": (N, 1920)}.
        dataset: Optional dataset object (unused, for protocol).
        module: Optional module object (unused, for protocol).
        k: Number of neighbors for Jaccard computation.
        metric: Distance metric for neighbor search.
        return_per_sample: If True, return per-sample scores instead of mean.
        return_pairwise: If True, return dict of pairwise Jaccard arrays.

    Returns:
        If return_pairwise: Dict mapping pair names to per-sample arrays.
        If return_per_sample: Array of shape (N,) with mean Jaccard per sample.
        Otherwise: Scalar mean Jaccard across all pairs and samples.
    """
    # Single array case
    if isinstance(embeddings, np.ndarray):
        n = _ensure_2d(embeddings).shape[0]
        if return_per_sample:
            return np.ones(n)  # Self-comparison is perfect
        return 1.0

    # Multi-modal case
    modality_names = list(embeddings.keys())
    n_modalities = len(modality_names)

    if n_modalities < 2:
        raise ValueError("Need at least 2 modalities for cross-modal comparison")

    # Validate sample counts
    n_samples = _ensure_2d(embeddings[modality_names[0]]).shape[0]
    for name, emb in embeddings.items():
        if _ensure_2d(emb).shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: {name}")

    # Compute pairwise Jaccard
    pairwise_results = {}
    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            pair_name = f"{modality_names[i]}_{modality_names[j]}"
            jaccard = cross_modal_jaccard_pairwise(
                embeddings[modality_names[i]],
                embeddings[modality_names[j]],
                k=k,
                metric=metric,
            )
            pairwise_results[pair_name] = jaccard

    if return_pairwise:
        return pairwise_results

    all_scores = np.stack(list(pairwise_results.values()), axis=0)
    per_sample_mean = all_scores.mean(axis=0)

    if return_per_sample:
        return per_sample_mean

    return float(per_sample_mean.mean())
