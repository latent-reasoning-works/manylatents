"""Diffusion Condensation metric for stable component detection."""

import logging
from typing import Any, Optional, Union

import graphtools
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from manylatents.metrics.registry import register_metric

logger = logging.getLogger(__name__)


def build_diffusion_operator(X: np.ndarray, knn: int = 5, decay: int = 40) -> np.ndarray:
    """Build row-stochastic diffusion operator from data using graphtools."""
    G = graphtools.Graph(X, knn=knn, decay=decay, n_jobs=1, verbose=0)
    K = np.array(G.K.todense())
    row_sums = np.maximum(K.sum(axis=1, keepdims=True), 1e-10)
    return K / row_sums


def run_condensation(
    X: np.ndarray,
    P: np.ndarray,
    scale: float = 1.025,
    granularity: float = 0.1,
    max_iterations: int = 500,
    n_subsample: int = 1000,
) -> tuple[list[int], np.ndarray]:
    """Run condensation loop, returning component counts per scale and gradient."""
    # Compute condensation params from subsampled distances
    n_sub = min(n_subsample, X.shape[0])
    X_sub = X[np.random.choice(X.shape[0], n_sub, replace=False)] if X.shape[0] > n_sub else X
    dists = pdist(X_sub)
    if len(dists) == 0:
        return [1], np.array([0])

    epsilon = np.std(dists) * granularity
    merge_threshold = np.percentile(dists, 0.1) + epsilon * 0.1

    # Initialize: each point is its own cluster
    clusters = np.arange(X.shape[0])
    X_current = X.copy()
    n_components = [len(np.unique(clusters))]

    for _ in range(max_iterations):
        unique = np.unique(clusters)
        n_clusters = len(unique)
        if n_clusters <= 1:
            break

        # Compute centroids and pairwise distances
        centroids = np.array([X_current[clusters == c].mean(axis=0) for c in unique])
        dist_matrix = squareform(pdist(centroids))
        np.fill_diagonal(dist_matrix, np.inf)
        merge_pairs = np.argwhere(dist_matrix < merge_threshold)

        if len(merge_pairs) == 0:
            # No merges possible - smooth and increase threshold
            X_current = P @ X_current
            merge_threshold *= scale
            n_components.append(n_clusters)
            continue

        # Union-find for transitive merging
        parent = {c: c for c in unique}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        for i, j in merge_pairs:
            if i < j:
                pi, pj = find(unique[i]), find(unique[j])
                if pi != pj:
                    parent[pi] = pj

        # Remap clusters to contiguous indices
        new_clusters = np.array([find(c) for c in clusters])
        remap = {old: new for new, old in enumerate(np.unique(new_clusters))}
        clusters = np.array([remap[c] for c in new_clusters])
        n_components.append(len(np.unique(clusters)))

    gradient = np.abs(np.diff(n_components)) if len(n_components) > 1 else np.array([0])
    return n_components, gradient


def find_stable_scales(n_components: list[int], gradient: np.ndarray) -> tuple[list[int], int]:
    """Find stable scales using numpy run-length encoding."""
    if len(gradient) == 0:
        return [0], n_components[0] if n_components else 1

    n_arr = np.array(n_components)

    # Run-length encoding via numpy
    change_idx = np.where(np.diff(n_arr) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_idx])
    run_ends = np.concatenate([change_idx, [len(n_arr)]])
    run_lengths = run_ends - run_starts
    run_values = n_arr[run_starts]

    # Filter to stable runs (length >= 2)
    stable_mask = run_lengths >= 2
    if not stable_mask.any():
        return [len(n_arr) - 1], int(n_arr[-1])

    stable_starts = run_starts[stable_mask]
    stable_lengths = run_lengths[stable_mask]
    stable_values = run_values[stable_mask]

    # Score runs: prefer longer, later, fewer components
    def score(start, length, value):
        position_bonus = 1 + start / len(n_arr)
        size_penalty = np.log(1 + value)
        return length * position_bonus / max(size_penalty, 0.1)

    scores = [score(s, l, v) for s, l, v in zip(stable_starts, stable_lengths, stable_values)]
    best_idx = int(np.argmax(scores))

    stable_scales = [int(s + l // 2) for s, l in zip(stable_starts, stable_lengths)]
    return stable_scales, int(stable_values[best_idx])


@register_metric(
    aliases=["diffusion_condensation"],
    default_params={"scale": 1.025, "granularity": 0.1, "knn": 5, "decay": 40, "n_pca": 50, "n_subsample": 1000, "output_mode": "stable"},
    description="Diffusion condensation score",
)
def DiffusionCondensation(
    embeddings: np.ndarray,
    dataset: Optional[Any] = None,
    module: Optional[Any] = None,
    scale: float = 1.025,
    granularity: float = 0.1,
    knn: int = 5,
    decay: int = 40,
    n_pca: Optional[int] = 50,
    n_subsample: int = 1000,
    output_mode: str = "stable",
) -> Union[int, dict[str, Any]]:
    """
    Compute component stability via diffusion condensation.

    Can be applied to any embedding to measure stable connected components (beta_0).

    Parameters:
        n_subsample: Max samples for distance computation (default 1000).
                     Increase for larger datasets if resolution matters.
    """
    if embeddings.shape[0] < 2:
        logger.warning("DiffusionCondensation: Too few points")
        return 1 if output_mode == "single" else {"n_stable_components": 1, "stable_scales": [0]}

    X = embeddings
    if n_pca and X.shape[1] > n_pca:
        X = PCA(n_components=n_pca).fit_transform(X)

    P = build_diffusion_operator(X, knn=knn, decay=decay)
    n_components, gradient = run_condensation(X, P, scale=scale, granularity=granularity, n_subsample=n_subsample)
    stable_scales, n_stable = find_stable_scales(n_components, gradient)

    logger.info(f"DiffusionCondensation: {len(n_components)} scales, {n_stable} stable components")

    if output_mode == "single":
        return n_stable

    result = {"n_stable_components": n_stable, "stable_scales": stable_scales}
    if output_mode == "stable":
        result["gradient_at_stable"] = [gradient[s] if s < len(gradient) else 0 for s in stable_scales]
    elif output_mode == "all":
        result["n_components_per_scale"] = np.array(n_components)
        result["gradient"] = gradient
    return result
