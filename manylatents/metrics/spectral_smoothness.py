"""Spectral smoothness of the input kNN graph: 1 - lambda_j of its normalised adjacency, for a chosen index j.

lambda_1 = 1 >= lambda_2 >= ... are the top eigenvalues of D^(-1/2) A D^(-1/2), where A is the symmetric binary k-NN
graph on its largest connected component. A d-dimensional Laplacian embedding uses lambda_2 ... lambda_(d+1), so
``eigen_index = d + 1`` asks how far the last direction of a d-dimensional embedding is from smooth. The default 3 is the
2D case: a small 1 - lambda_3 means a smooth two-dimensional structure dominates the graph. In a held-out test on 27 data
sets it predicted how much UMAP and t-SNE layouts change between seeds (Spearman -0.88 and -0.91), beyond intrinsic
dimension and hubness. Input-only: computed before any layout.
"""
from typing import Optional

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.knn import compute_knn


@register_metric(
    aliases=["spectral_smoothness", "smooth2d"],
    default_params={"eigen_index": 3, "n_neighbors": 15},
    description="1 - lambda_j of the normalised k-NN graph; j = d + 1 for a d-dimensional embedding (default 3, the 2D case)",
)
def SpectralSmoothness(
    embeddings: np.ndarray,
    dataset: Optional[object] = None,
    module: Optional[LatentModule] = None,
    eigen_index: int = 3,
    n_neighbors: int = 15,
    cache: Optional[dict] = None,
) -> float:
    """Return 1 - lambda_(eigen_index) of the normalised symmetric k-NN graph of ``embeddings`` (largest component).

    ``eigen_index`` is 1-based: lambda_1 = 1 is the trivial eigenvalue, so it must be at least 2. With ``at: dataset`` the
    high-dimensional input is passed as ``embeddings``, so the value is input-only.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    if isinstance(eigen_index, bool) or not isinstance(eigen_index, (int, np.integer)) or eigen_index < 2:
        raise ValueError("eigen_index must be an integer >= 2 (lambda_1 = 1 is trivial)")
    X = np.ascontiguousarray(np.asarray(embeddings), dtype=np.float32)
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    _, idx = compute_knn(X, k=k, include_self=False, cache=cache)
    A = sp.csr_matrix((np.ones(idx.size), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)
    _, labels = connected_components(A, directed=False)
    keep = np.flatnonzero(labels == np.bincount(labels).argmax())
    if eigen_index >= len(keep):
        raise ValueError(f"eigen_index {eigen_index} needs more than {eigen_index} points in the largest component (has {len(keep)})")
    A = A[keep][:, keep]
    d = np.asarray(A.sum(axis=1)).ravel()
    inv = sp.diags(1.0 / np.sqrt(d))
    v0 = np.random.default_rng(0).uniform(0.5, 1.5, size=len(keep))      # fixed ARPACK start: deterministic output
    lam = np.sort(eigsh(inv @ A @ inv, k=int(eigen_index), which="LA", v0=v0, tol=1e-12, return_eigenvectors=False))[::-1]
    return float(1.0 - lam[eigen_index - 1])
