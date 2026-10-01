import logging
from typing import Optional, Tuple

import numpy as np

from manylatents.utils.exceptions import MeasurementUnavailable

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
    h.update(f"knn-self-v2{data.shape}{data.dtype}".encode())
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

    Backends can differ in floating-point distances and neighbor tie-breaking,
    so switching backends can change derived measurements.

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

    n_samples = data.shape[0]
    if (not isinstance(k, (int, np.integer)) or isinstance(k, bool)
            or not 0 < k < n_samples):
        raise MeasurementUnavailable(
            f"kNN requires 0 < k < n_samples; got k={k}, n_samples={n_samples}"
        )
    # Typed backends such as FAISS require a Python int. Normalize before
    # arithmetic/cache lookup so numpy integer counts use the same backend.
    k = int(k)

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
            logger.warning(
                f"FAISS failed ({type(e).__name__}: {e}), falling back to sklearn; "
                "neighbors and distances may differ between backends, changing "
                "derived measurements"
            )
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, indices = nbrs.kneighbors(data)
        logger.info(f"compute_knn: sklearn, n={data.shape[0]}, d={data.shape[1]}, k={k}")

    # Ties may put self anywhere, or outside the returned k+1 candidates.
    # Canonicalize before caching: self first, then k actual other indices.
    other = indices != np.arange(n_samples)[:, None]
    order = np.argsort(~other, axis=1, kind="stable")[:, :k]
    indices = np.column_stack((np.arange(n_samples), np.take_along_axis(indices, order, axis=1)))
    distances = np.column_stack((np.zeros(n_samples), np.take_along_axis(distances, order, axis=1)))

    # Store in cache (always with self included)
    if cache is not None:
        cache[_content_key(data)] = (k, distances, indices)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


def _as_float32_matrix(x, name: str) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if (
        x.ndim != 2 or 0 in x.shape
        or not np.issubdtype(x.dtype, np.number)
        or np.iscomplexobj(x)
    ):
        raise MeasurementUnavailable(f"{name} must be a nonempty real numeric matrix")
    if not np.isfinite(x).all():
        raise MeasurementUnavailable(f"{name} contains non-finite values")
    return np.ascontiguousarray(x, dtype=np.float32)


def compute_knn_query(
    reference: np.ndarray,
    query: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """k nearest ``reference`` rows for each ``query`` row (FAISS, else sklearn).

    Unlike :func:`compute_knn`, the two point sets are different, so there is
    no self to exclude and nothing is cached. A query row that coincides with a
    reference row gets that row back as its nearest neighbour; FAISS may report
    a small non-zero distance for it, so callers that must exclude coincident
    rows identify them by content, not by ``distance == 0``.

    Args:
        reference: (n_reference, n_features) point set that is searched.
        query: (n_query, n_features) points whose neighbours are wanted.
        k: number of neighbours, ``0 < k <= n_reference``.

    Returns:
        (distances, indices), both (n_query, k). Distances are Euclidean,
        ascending, float64. Indices are int64 row numbers into ``reference``.
    """
    reference = _as_float32_matrix(reference, "reference")
    query = _as_float32_matrix(query, "query")
    if reference.shape[1] != query.shape[1]:
        raise MeasurementUnavailable(
            f"reference has {reference.shape[1]} features, query has {query.shape[1]}"
        )
    n_reference = reference.shape[0]
    if (not isinstance(k, (int, np.integer)) or isinstance(k, bool)
            or not 0 < k <= n_reference):
        raise MeasurementUnavailable(
            f"query kNN requires 0 < k <= n_reference; got k={k}, n_reference={n_reference}"
        )
    k = int(k)

    try:
        import faiss

        index = faiss.IndexFlatL2(reference.shape[1])
        backend = "faiss-cpu"
        if getattr(faiss, "get_num_gpus", lambda: 0)() > 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                backend = "faiss-gpu"
            except Exception:
                pass
        index.add(reference)
        distances, indices = index.search(query, k)
        distances = np.sqrt(np.maximum(distances, 0))
        logger.info(
            f"compute_knn_query: {backend}, n_reference={n_reference}, "
            f"n_query={query.shape[0]}, k={k}"
        )
    except Exception as e:
        if not isinstance(e, ImportError):
            logger.warning(
                f"FAISS failed ({type(e).__name__}: {e}), falling back to sklearn; "
                "neighbors and distances may differ between backends, changing "
                "derived measurements"
            )
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k).fit(reference)
        distances, indices = nbrs.kneighbors(query)
        logger.info(
            f"compute_knn_query: sklearn, n_reference={n_reference}, "
            f"n_query={query.shape[0]}, k={k}"
        )

    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)
