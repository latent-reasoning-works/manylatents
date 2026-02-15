import copy
import logging
from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, ListConfig
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger(__name__)


def _svd_gpu(
    embeddings: np.ndarray,
    idx: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Compute batched SVD on GPU via torch.linalg.svdvals.

    Args:
        embeddings: (n_samples, d) float32 array.
        idx: (n_samples, k) neighbor indices (self excluded).
        chunk_size: Number of samples per chunk.

    Returns:
        Singular values array (n_samples, min(k, d)).
    """
    import torch

    device = torch.device("cuda")
    emb_t = torch.from_numpy(embeddings).to(device)
    idx_t = torch.from_numpy(idx).long().to(device)
    n_samples = idx.shape[0]

    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = emb_t[idx_t[start:end]]  # (chunk, k, d)
        centered = neigh - neigh.mean(dim=1, keepdim=True)
        s = torch.linalg.svdvals(centered)  # (chunk, min(k, d))
        sv_chunks.append(s.cpu().numpy())

    return np.concatenate(sv_chunks, axis=0)


def _svd_cpu(
    embeddings: np.ndarray,
    idx: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Compute batched SVD on CPU via np.linalg.svd.

    Args:
        embeddings: (n_samples, d) float32 array.
        idx: (n_samples, k) neighbor indices (self excluded).
        chunk_size: Number of samples per chunk.

    Returns:
        Singular values array (n_samples, min(k, d)).
    """
    n_samples = idx.shape[0]
    sv_chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        neigh = embeddings[idx[start:end]]  # (chunk, k, d)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        s = np.linalg.svd(centered, compute_uv=False)  # (chunk, min(k, d))
        sv_chunks.append(s)
    return np.concatenate(sv_chunks, axis=0)


def compute_svd_cache(
    embeddings: np.ndarray,
    knn_indices: np.ndarray,
    k_values: set,
) -> Dict[int, np.ndarray]:
    """Compute SVD of centered kNN neighborhoods once for shared use across metrics.

    Automatically selects the fastest available backend:
    1. torch GPU if CUDA is available (~10-50x faster than CPU for large batches)
    2. numpy CPU as fallback

    For each k in k_values, gathers k neighbors per sample, centers them,
    and computes singular values via batch SVD.

    Args:
        embeddings: (n_samples, d) array.
        knn_indices: (n_samples, max_k+1) indices from kNN cache (index 0 is self).
        k_values: Set of k values to compute SVD for.

    Returns:
        Dict mapping k -> singular values array of shape (n_samples, min(k, d)).
    """
    # Detect GPU availability once
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
    except ImportError:
        pass

    backend = "torch-gpu" if use_gpu else "numpy-cpu"
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n_samples, d = embeddings.shape
    result = {}

    for k in sorted(k_values):
        idx = knn_indices[:, 1:k + 1]  # (n_samples, k), skip self
        chunk_size = max(1, min(10_000, int(2e9 / (k * d * 4))))

        if use_gpu:
            try:
                result[k] = _svd_gpu(embeddings, idx, chunk_size)
            except Exception as e:
                logger.warning(f"GPU SVD failed ({type(e).__name__}: {e}), falling back to CPU")
                result[k] = _svd_cpu(embeddings, idx, chunk_size)
                backend = "numpy-cpu (gpu-fallback)"
        else:
            result[k] = _svd_cpu(embeddings, idx, chunk_size)

        logger.info(
            f"compute_svd_cache: {backend}, k={k}, shape={result[k].shape}"
        )

    return result


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
    # Check cache for a usable superset
    if cache is not None:
        key = id(data)
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
        cache[id(data)] = (k, distances, indices)

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


def compute_eigenvalues(
    module,
    cache: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Compute sorted eigenvalues of module's symmetric affinity matrix.

    Args:
        module: Fitted LatentModule with affinity_matrix() method, or None.
        cache: Optional dict. Stores result under key "eigenvalues".

    Returns:
        Eigenvalues sorted descending, or None if unavailable.
    """
    # Check cache first — pre-warmed cache works even without a module
    if cache is not None and "eigenvalues" in cache:
        return cache["eigenvalues"]

    if module is None:
        return None

    try:
        A = module.affinity_matrix(use_symmetric=True)
    except (NotImplementedError, AttributeError):
        return None

    eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]

    if cache is not None:
        cache["eigenvalues"] = eigenvalues

    return eigenvalues


def flatten_and_unroll_metrics(
    all_metrics: DictConfig
) -> Dict[str, DictConfig]:
    """
    Walks through every subgroup under cfg.metrics (dataset, embedding, module),
    finds each DictConfig with a _target_, and for any keys whose value is
    a list/ListConfig, builds one sub-config per element (Cartesian‐product
    if multiple list-valued keys are present).

    Returns:
      name→single‐value DictConfig, where name is
        "{group}.{metric}" or
        "{group}.{metric}__{param1}_{v1}_{param2}_{v2}…"
    """
    flat: Dict[str, DictConfig] = {}

    for group_name, group_cfg in all_metrics.items():
        if not isinstance(group_cfg, DictConfig):
            continue

        for metric_name, metric_cfg in group_cfg.items():
            if not (isinstance(metric_cfg, DictConfig) and "_target_" in metric_cfg):
                continue

            # 1) collect all keys that are list-valued
            sweep_keys = []
            sweep_vals = []
            for k, v in metric_cfg.items():
                if isinstance(v, (list, tuple, ListConfig)):
                    sweep_keys.append(k)
                    sweep_vals.append(list(v))

            # 2) if no sweep, just copy as-is
            if not sweep_keys:
                flat[f"{group_name}.{metric_name}"] = metric_cfg
                continue

            # 3) cartesian‐product over all sweep values
            for combo in product(*sweep_vals):
                cfg_copy = copy.deepcopy(metric_cfg)
                suffix_parts = []
                for k, val in zip(sweep_keys, combo):
                    # coerce to native types
                    if isinstance(val, ListConfig): val = list(val)
                    if isinstance(val, float) and float(val).is_integer():
                        val = int(val)
                    setattr(cfg_copy, k, val)
                    suffix_parts.append(f"{k}_{val}")

                flat_name = f"{group_name}.{metric_name}__{'_'.join(suffix_parts)}"
                flat[flat_name] = cfg_copy

    return flat
###### METRIC-SPECIFIC HELPERS
def haversine_vectorized(coords):
    """
    Compute pairwise haversine distances in a vectorized manner.
    coords: (n_samples, 2) array of [latitude, longitude] in radians.
    Returns a (n_samples x n_samples) distance matrix.
    """
    lat = coords[:, 0][:, np.newaxis]
    lon = coords[:, 1][:, np.newaxis]

    dlat = lat - lat.T
    dlon = lon - lon.T

    a = np.sin(dlat / 2.0)**2 + np.cos(lat)*np.cos(lat.T)*np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth radius ~6371 km
    distances = 6371.0 * c
    return distances


def compute_geodesic_distances(embedding, k=10, metric='euclidean', verbose=0):
    """
    Build a KNN graph from 'embedding' (mode='distance') and compute
    shortest-path distances (Dijkstra). 
    Returns a flattened (upper-tri) array or None if disconnected.
    """
    knn_graph = kneighbors_graph(
        embedding, n_neighbors=k, mode='distance',
        metric=metric, include_self=False
    )
    n_components, labels = connected_components(knn_graph, directed=False)
    if n_components > 1:
        if verbose:
            print(f"Disconnected KNN graph (n_components={n_components}). Returning None.")
        return None

    dist_matrix = shortest_path(knn_graph, directed=False, method='D')
    flattened_dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    return flattened_dists


def compute_knn_laplacian(ancestry_coords, k=5, normalized=True, verbose=0):
    """
    Build a KNN connectivity graph and compute the Laplacian (normalized or not).
    Returns an (N x N) Laplacian array.
    """
    adjacency = kneighbors_graph(
        ancestry_coords, n_neighbors=k, mode='connectivity', include_self=False
    ).toarray()

    degree_matrix = np.diag(adjacency.sum(axis=1))

    if normalized:
        # L_norm = I - D^-1/2 * A * D^-1/2
        with np.errstate(divide='ignore'):
            deg_inv_sqrt = np.linalg.pinv(np.sqrt(degree_matrix))
        laplacian = np.eye(adjacency.shape[0]) - deg_inv_sqrt @ adjacency @ deg_inv_sqrt
    else:
        # L = D - A
        laplacian = degree_matrix - adjacency

    return laplacian


def compute_average_smoothness(laplacian, admixture_ratios):
    """
    For each admixture component, compute x^T L x, then return average across all.
    """
    smooth_vals = []
    for i in range(admixture_ratios.shape[1]):
        x = admixture_ratios[:, i]
        val = x.T @ laplacian @ x
        smooth_vals.append(val)
    return np.mean(smooth_vals)
