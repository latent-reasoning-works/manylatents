import warnings
from typing import Optional

import numpy as np
import networkx as nx
from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.metrics.registry import register_metric
from manylatents.utils.metrics import resolve_matrix

def connected_components(kernel_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the sizes of connected components from a kernel/affinity matrix.

    Parameters:
      - kernel_matrix: The kernel or affinity matrix representing graph connectivity.

    Returns:
      - Array of component sizes, sorted in descending order.
    """
    graph = nx.from_numpy_array(kernel_matrix)  # Convert adjacency matrix to a graph
    component_sizes = np.sort(np.array([len(k) for k in nx.connected_components(graph)]))[::-1]

    return component_sizes

##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

@register_metric(
    aliases=["connected_components"],
    default_params={"ignore_diagonal": False, "matrix_source": "kernel"},
    description="Number of connected components in the kNN graph",
)
def ConnectedComponents(dataset, embeddings: np.ndarray, module: LatentModule, ignore_diagonal: bool = False, matrix_source: str = "kernel", cache: Optional[dict] = None) -> np.ndarray:
    """
    Compute connected components from the module's graph matrix.

    Args:
        dataset: Dataset object (unused).
        embeddings: Low-dimensional embeddings (unused).
        module: LatentModule instance.
        ignore_diagonal: Whether to ignore diagonal when getting the matrix.
        matrix_source: Which matrix to use: "kernel", "affinity", or "adjacency".

    Returns:
        `(n_components, component_sizes)` — the scalar the name promises, plus the raw sizes.
        `(nan, [nan])` if the matrix is not available.

    This returned only the component SIZES array, which `_to_scalar` then averaged: a graph
    with sizes [67, 67, 66] recorded **66.667** under a description reading "Number of
    connected components". Worse, the recorded value grew as the true count fell, so it was
    anti-correlated with the thing it claimed to measure. The `(scalar, raw)` tuple is the
    convention the registry already handles.
    """
    try:
        mat = resolve_matrix(module, source=matrix_source, ignore_diagonal=ignore_diagonal)
    except (NotImplementedError, AttributeError):
        warnings.warn(
            f"ConnectedComponents metric skipped: {module.__class__.__name__} "
            f"does not expose a {matrix_source}_matrix.",
            RuntimeWarning
        )
        return float("nan"), np.array([np.nan])

    component_sizes = connected_components(kernel_matrix=mat)
    return float(len(component_sizes)), component_sizes
