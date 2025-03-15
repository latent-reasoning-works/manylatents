import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import kneighbors_graph


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
