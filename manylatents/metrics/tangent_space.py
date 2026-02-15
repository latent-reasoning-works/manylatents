import logging
from typing import Optional, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import ColormapInfo
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import compute_knn

logger = logging.getLogger(__name__)


@register_metric(
    aliases=["tangent_space"],
    default_params={"n_neighbors": 25, "variance_threshold": 0.95},
    description="Tangent space alignment between original and embedded spaces",
)
def TangentSpaceApproximation(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    n_neighbors: int = 25,
    variance_threshold: float = 0.95,
    return_per_sample: bool = False,
    cache: Optional[dict] = None,
) -> Union[float, Tuple[np.ndarray, ColormapInfo]]:
    """
    Estimate the local manifold dimension by approximating the tangent space via PCA.

    Args:
        embeddings:   (n_samples, n_features) array.
        dataset:      (unused) kept for Protocol compatibility.
        module:       (unused) kept for Protocol compatibility.
        n_neighbors:  how many neighbors to use for local PCA.
        variance_threshold: cumulative variance threshold to determine local dim.
        return_per_sample: if True, return per-sample dimensions with viz metadata.
        cache: Optional shared cache dict. Passed through to compute_knn().

    Returns:
        float: Average local dimension (if return_per_sample=False)
        Tuple[np.ndarray, ColormapInfo]: Per-sample dimensions with visualization
            metadata (if return_per_sample=True). The ColormapInfo specifies
            categorical rendering with dynamic label generation.
    """
    n_samples = embeddings.shape[0]

    _, indices = compute_knn(embeddings, k=n_neighbors, include_self=True, cache=cache)

    k = indices.shape[1] - 1
    chunk_size = max(1, min(10_000, int(2e9 / (k * embeddings.shape[1] * 4))))

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[indices[start:end, 1:]]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        sv_chunks.append(np.linalg.svd(centered, compute_uv=False))

    s = np.concatenate(sv_chunks, axis=0)

    # Compute local dimensions from singular values
    s2 = s * s
    total_var = s2.sum(axis=1, keepdims=True)
    total_var = np.where(total_var > 0, total_var, 1.0)
    var_explained = s2 / total_var
    cum_var = np.cumsum(var_explained, axis=1)
    dims_array = (cum_var < variance_threshold).sum(axis=1) + 1

    if return_per_sample:
        logger.info(f"TangentSpaceApproximation: Per-sample local dimensions: {dims_array}")

        viz_info = ColormapInfo(
            cmap="categorical",
            label_names=None,
            label_format="Dim = {}",
            is_categorical=True,
        )
        return dims_array, viz_info
    else:
        avg_dim = float(np.mean(dims_array))
        logger.info(f"TangentSpaceApproximation: Average local dimension computed as {avg_dim}")
        return avg_dim
