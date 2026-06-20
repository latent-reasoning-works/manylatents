"""Input-geometry diagnostics as registered metrics.

Fold: per-point folding of the input manifold (disagreement between a point's
geodesic neighbors and its straight-line/ambient neighbors), an input-only
quantity that forecasts where an embedding will fail to preserve fine structure.
PreservationScale: the finest neighborhood at which an embedding's neighbors
begin to match the manifold's own (geodesic) neighbors.

STAGED COPY (2026-06-19): the manylatents working clone is git-corrupted; this is
the canonical, complete source for both metrics, to be dropped into a clean
manylatents clone at `manylatents/metrics/preservation_diagnostics.py`. See APPLY.md.
"""
from __future__ import annotations
import numpy as np
from manylatents.metrics.registry import register_metric


def _graph_geodesic(X, k_graph):
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path, connected_components
    X = np.asarray(X, np.float64)
    G = kneighbors_graph(X, n_neighbors=int(k_graph), mode="distance", include_self=False)
    G = G.maximum(G.T)
    n_comp, labels = connected_components(G, directed=False)
    if n_comp > 1:
        keep = np.where(labels == int(np.bincount(labels).argmax()))[0]
        G = G[keep][:, keep]
    else:
        keep = np.arange(X.shape[0])
    D = shortest_path(G, method="D", directed=False)
    return np.asarray(D, np.float64), keep


def _ladder(k_min, k_max, n_steps):
    g = np.logspace(np.log10(k_min), np.log10(k_max), int(n_steps))
    return np.unique(np.round(g)).astype(int)


def _recall_curve(D_a, D_b, k_grid):
    D_a = np.asarray(D_a, np.float64)
    D_b = np.asarray(D_b, np.float64)
    n = D_a.shape[0]
    k_grid = np.asarray(k_grid, int)
    kmax = int(k_grid.max())
    oa = np.argsort(D_a, axis=1)[:, 1:kmax + 1]
    ob = np.argsort(D_b, axis=1)[:, 1:kmax + 1]
    rec = np.empty((n, len(k_grid)))
    for i in range(n):
        ai, bi = oa[i], ob[i]
        for t, k in enumerate(k_grid):
            rec[i, t] = np.intersect1d(ai[:k], bi[:k], assume_unique=True).size / k
    return rec


def _preservation_scale(recall, k_grid, tau):
    recall = np.asarray(recall, np.float64)
    k_grid = np.asarray(k_grid)
    n = recall.shape[0]
    s = np.full(n, float(k_grid[-1]))
    for i in range(n):
        hit = np.where(recall[i] >= tau)[0]
        if hit.size:
            s[i] = float(k_grid[hit[0]])
    return s


def _fold(D_geo, D_amb, k_ref):
    rec = _recall_curve(D_geo, D_amb, np.array([int(k_ref)]))
    return 1.0 - rec[:, 0]


@register_metric(
    aliases=["fold", "fold_obstruction"],
    default_params={"k_ref": 30, "k_graph": 8},
    description="Per-point input folding: 1 - recall(geodesic-kNN, ambient-kNN). "
                "High where the manifold folds (ambient neighbors are geodesically distant). Input-only.",
)
def Fold(embeddings, dataset=None, module=None, k_ref=30, k_graph=8, cache=None) -> dict:
    from scipy.spatial.distance import cdist
    X = np.asarray(dataset.data, np.float64)
    D_geo, keep = _graph_geodesic(X, k_graph)
    fold_k = _fold(D_geo, cdist(X[keep], X[keep]), k_ref)
    fold = np.full(X.shape[0], np.nan)
    fold[keep] = fold_k
    return {"mean_fold": float(np.nanmean(fold)), "fold": fold}


@register_metric(
    aliases=["preservation_scale", "s_star"],
    default_params={"tau": 0.5, "k_graph": 8, "k_max": 200, "k_min": 2, "n_steps": 20},
    description="Per-point preservation scale s*: finest neighborhood size at which the "
                "embedding's kNN recall against the input geodesic kNN reaches tau.",
)
def PreservationScale(embeddings, dataset=None, module=None, tau=0.5, k_graph=8,
                      k_max=200, k_min=2, n_steps=20, cache=None) -> dict:
    from scipy.spatial.distance import cdist
    X = np.asarray(dataset.data, np.float64)
    Y = np.asarray(embeddings, np.float64)
    D_geo, keep = _graph_geodesic(X, k_graph)
    ladder = _ladder(k_min, k_max, n_steps)
    rec = _recall_curve(D_geo, cdist(Y[keep], Y[keep]), ladder)
    s_k = _preservation_scale(rec, ladder, tau)
    s = np.full(X.shape[0], np.nan)
    s[keep] = s_k
    return {"mean_s_star": float(np.nanmean(s)), "s_star": s}
