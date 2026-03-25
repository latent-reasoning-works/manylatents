import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _content_key(data) -> str:
    """O(1) content hash: shape + dtype + first/last row bytes.

    Always normalizes to float32 contiguous so the hash is consistent
    regardless of whether the caller passes the original or converted data.
    """
    import hashlib
    import torch
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.ascontiguousarray(data, dtype=np.float32)
    h = hashlib.sha256()
    h.update(f"{data.shape}{data.dtype}".encode())
    h.update(data[0].tobytes())
    h.update(data[-1].tobytes())
    return h.hexdigest()[:16]


def compute_knn(
    data: np.ndarray,
    k: int,
    include_self: bool = True,
    cache: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbors using FAISS-GPU > FAISS-CPU > sklearn.

    Automatically selects the fastest available backend:
    1. FAISS-GPU if a CUDA device is available (~100x faster than CPU)
    2. FAISS-CPU if faiss is installed (~10-50x faster than sklearn)
    3. sklearn NearestNeighbors as fallback

    Args:
        data: (n_samples, n_features) float32 array.
        k: Number of neighbors (excluding self).
        include_self: If True, returns k+1 columns with self at index 0.
            If False, returns k columns (self excluded).
        cache: Optional dict for caching. Keyed by id(data).
            If a cached result exists with k >= requested k, slices and returns.
            Otherwise computes and stores the result.

    Returns:
        (distances, indices) — both shape (n_samples, k+1) if include_self,
        or (n_samples, k) if not.
    """
    # Ensure numpy (LatentModule.transform() returns tensors)
    import torch
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Default k when None (e.g. from ${neighborhood_size} resolving to null)
    if k is None:
        k = 15

    # Check cache for a usable superset
    if cache is not None:
        key = _content_key(data)
        if key in cache:
            cached_k, cached_dists, cached_idxs = cache[key]
            if cached_k >= k:
                n = k + 1  # cached always includes self
                dists, idxs = cached_dists[:, :n], cached_idxs[:, :n]
                if not include_self:
                    dists, idxs = dists[:, 1:], idxs[:, 1:]
                return dists, idxs

    n_samples = data.shape[0]
    if k >= n_samples:
        import warnings
        warnings.warn(
            f"Clamping k from {k} to {n_samples - 1} (n_samples={n_samples})",
            UserWarning,
        )
        k = n_samples - 1
    if k <= 0:
        return np.zeros((n_samples, 0)), np.zeros((n_samples, 0), dtype=np.int64)

    data = np.ascontiguousarray(data, dtype=np.float32)
    n_neighbors = k + 1  # always query k+1 to include self, then trim

    try:
        import faiss

        d = data.shape[1]
        index = faiss.IndexFlatL2(d)

        # Try GPU if available (faiss-gpu-cu12 package)
        backend = "faiss-cpu"
        if getattr(faiss, "get_num_gpus", lambda: 0)() > 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                backend = "faiss-gpu"
            except Exception:
                pass

        index.add(data)
        distances, indices = index.search(data, n_neighbors)
        # FAISS returns squared L2; convert to Euclidean for sklearn compat
        distances = np.sqrt(np.maximum(distances, 0))
        logger.info(f"compute_knn: {backend}, n={data.shape[0]}, d={d}, k={k}")
    except Exception as e:
        # Catches ImportError (no faiss), AttributeError (faiss-gpu without CUDA
        # runtime — module loads but symbols like IndexFlatL2 are missing), etc.
        if not isinstance(e, ImportError):
            logger.warning(f"FAISS failed ({type(e).__name__}: {e}), falling back to sklearn")
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, indices = nbrs.kneighbors(data)
        logger.info(f"compute_knn: sklearn, n={data.shape[0]}, d={data.shape[1]}, k={k}")

    # Store in cache (always with self included)
    if cache is not None:
        cache[_content_key(data)] = (k, distances, indices)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices
