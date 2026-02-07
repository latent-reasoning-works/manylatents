import copy
import logging
from itertools import product
from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig, ListConfig
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger(__name__)


def compute_knn(
    data: np.ndarray,
    k: int,
    include_self: bool = True,
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

    Returns:
        (distances, indices) — both shape (n_samples, k+1) if include_self,
        or (n_samples, k) if not.
    """
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

    if not include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    return distances, indices


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
