import warnings
import numpy as np
import networkx as nx
from manylatents.algorithms.latent.latent_module_base import LatentModule

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

def ConnectedComponents(dataset, embeddings: np.ndarray, module: LatentModule, ignore_diagonal: bool = False) -> np.ndarray:
    """
    Compute connected components from the module's kernel matrix.

    Args:
        dataset: Dataset object (unused).
        embeddings: Low-dimensional embeddings (unused).
        module: LatentModule instance with kernel_matrix method.
        ignore_diagonal: Whether to ignore diagonal when getting kernel matrix.

    Returns:
        Array of component sizes, or [nan] if kernel matrix not available.
    """
    try:
        kernel_mat = module.kernel_matrix(ignore_diagonal=ignore_diagonal)
    except (NotImplementedError, AttributeError):
        warnings.warn(
            f"ConnectedComponents metric skipped: {module.__class__.__name__} does not expose a kernel_matrix.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return connected_components(kernel_matrix=kernel_mat)
