"""Source-entry fragments that many latent modules share.

A source entry (an entry of a module's ``PARAMS``) holds only what the constructor signature
cannot give: bounds, ``sensible``, ``ladder``, ``limits``, ``role``, ``precomputed``, ``cost``
and the texts. It never holds ``default``. Each function returns a new plain dict, so a module
may add keys to it.
"""
from __future__ import annotations

from typing import Any

# numpy.random.RandomState (reached through sklearn.utils.check_random_state) accepts
# None or an int in [0, 2**32 - 1].
NUMPY_SEED_MAX = 2**32 - 1


def seed(meaning: str = "Seed of the random number generators.", numpy: bool = True) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "int", "also": [None], "role": "fixed", "meaning": meaning}
    if numpy:
        entry.update(min=0, max=NUMPY_SEED_MAX)
    return entry


def n_jobs(also_none: bool = True) -> dict[str, Any]:
    return {
        "type": "int", "also": [None] if also_none else [], "role": "infra",
        "meaning": "Number of parallel jobs; -1 uses every CPU.",
    }


def verbose() -> dict[str, Any]:
    return {"type": "bool", "role": "infra", "meaning": "Print progress messages."}


def device(meaning: str = "Device; None or 'auto' picks CUDA when it is available.") -> dict[str, Any]:
    # manylatents.utils.backend.resolve_device passes any other text to torch unchanged.
    return {"type": "str", "also": [None], "role": "infra", "meaning": meaning}


def backend(choices: list[str], meaning: str) -> dict[str, Any]:
    return {"type": "choice", "choices": choices, "also": [None], "role": "infra", "meaning": meaning}


def unused(meaning: str, type_: str | None = None) -> dict[str, Any]:
    """A parameter that is stored but changes nothing that the fit computes."""
    entry: dict[str, Any] = {"role": "infra", "meaning": meaning}
    if type_ is not None:
        entry["type"] = type_
    return entry


def fit_fraction() -> dict[str, Any]:
    # No library check: fit uses x[:max(1, int(fit_fraction * n))], so a value above 1 acts as 1
    # and a value at or below 1/n acts as one row. The bounds state the fraction it means.
    return {
        "type": "float", "min": 0, "min_open": True, "max": 1, "role": "knob",
        "cost": {"time": "up", "memory": "up", "note": "fit runs on this fraction of the rows"},
        "meaning": "Fraction of the input rows, taken from the start, used to fit.",
        "more": "Fits on more rows; slower, closer to the full-data fit.",
        "less": "Fits on fewer rows; faster.",
    }


def neighbours(meaning: str, more: str, less: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "int", "min": 1, "role": "knob",
        "cost": {"time": "up", "memory": "up", "note": "more neighbours per point"},
        "meaning": meaning, "more": more, "less": less,
    }
    entry.update(extra)
    return entry


def iterations(meaning: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "int", "role": "knob",
        "cost": {"time": "up", "memory": "none", "note": "more iterations"},
        "meaning": meaning,
        "more": "Runs longer; closer to convergence.",
        "less": "Stops earlier; may not converge.",
    }
    entry.update(extra)
    return entry


def tolerance(meaning: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "float", "role": "knob",
        "cost": {"time": "down", "memory": "none", "note": "a looser tolerance stops sooner"},
        "meaning": meaning,
        "more": "Looser convergence; stops sooner.",
        "less": "Stricter convergence; more iterations.",
    }
    entry.update(extra)
    return entry


# Names each library accepts for a distance. Lists, not live lookups, so a declaration is plain
# data and does not change silently when a library changes; tests compare them to the libraries.

# scikit-learn 1.8.0: Isomap._parameter_constraints["metric"] (a StrOptions).
ISOMAP_METRICS = [
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
    "euclidean", "hamming", "haversine", "jaccard", "l1", "l2", "mahalanobis", "manhattan",
    "matching", "minkowski", "nan_euclidean", "precomputed", "rogerstanimoto", "russellrao",
    "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "wminkowski", "yule",
]

# phate 2.0.0: PHATE._check_params, utils.check_in(..., knn_dist=...).
PHATE_DISTANCES = [
    "euclidean", "precomputed", "cosine", "correlation", "cityblock", "l1", "l2", "manhattan",
    "braycurtis", "canberra", "chebyshev", "dice", "hamming", "jaccard", "kulsinski",
    "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
    "sokalmichener", "sokalsneath", "sqeuclidean", "yule", "precomputed_affinity",
    "precomputed_distance",
]

# scipy 1.14.1: scipy.spatial.distance._METRICS_NAMES (the names pdist accepts).
PDIST_METRICS = [
    "braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
    "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
    "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
    "sqeuclidean", "yule",
]

# umap-learn 0.5.11: UMAP._validate_parameters accepts umap.distances.named_distances,
# pynndescent.distances.named_distances (pynndescent 0.6.0) and "precomputed".
UMAP_METRICS = [
    "bit_hamming", "bit_jaccard", "braycurtis", "canberra", "categorical", "chebyshev",
    "circular_kantorovich", "circular_wasserstein", "correlation", "cosine", "count", "dice",
    "dot", "euclidean", "hamming", "haversine", "hellinger", "hierarchical_categorical",
    "inner_product", "jaccard", "jensen-shannon", "jensen_shannon", "kantorovich",
    "kantorovich-1d", "kantorovich_1d", "kulsinski", "l1", "l2", "linf", "linfinity", "linfty",
    "ll_dirichlet", "mahalanobis", "manhattan", "matching", "minkowski", "ordinal", "poincare",
    "precomputed", "rogerstanimoto", "russellrao", "seuclidean", "sinkhorn", "sokalmichener",
    "sokalsneath", "spearmanr", "sqeuclidean", "standardised_euclidean", "string",
    "symmetric-kl", "symmetric_kl", "symmetric_kullback_liebler", "taxicab", "true_angular",
    "tsss", "wasserstein", "wasserstein-1d", "wasserstein_1d", "weighted_minkowski",
    "wminkowski", "yule",
]

# openTSNE 1.0.4 with method="approx": "precomputed", Annoy.VALID_METRICS, and, through
# NNDescent.check_metric, NNDescent.VALID_METRICS and pynndescent.distances.named_distances.
OPENTSNE_METRICS = [
    "bit_hamming", "bit_jaccard", "braycurtis", "canberra", "chebyshev", "circular_kantorovich",
    "circular_wasserstein", "correlation", "cosine", "dice", "dot", "euclidean", "hamming",
    "haversine", "hellinger", "inner_product", "jaccard", "jensen-shannon", "jensen_shannon",
    "kantorovich", "kantorovich-1d", "kantorovich_1d", "kulsinski", "l1", "l2", "linf",
    "linfinity", "linfty", "mahalanobis", "manhattan", "matching", "minkowski", "precomputed",
    "rogerstanimoto", "russellrao", "seuclidean", "sinkhorn", "sokalmichener", "sokalsneath",
    "spearmanr", "sqeuclidean", "standardised_euclidean", "symmetric-kl", "symmetric_kl",
    "symmetric_kullback_liebler", "taxicab", "true_angular", "tsss", "wasserstein",
    "wasserstein-1d", "wasserstein_1d", "weighted_minkowski", "wminkowski", "yule",
]
