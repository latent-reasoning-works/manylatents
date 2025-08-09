import warnings
import numpy as np
import networkx as nx
from manylatents.algorithms.latent_module_base import LatentModule

def connected_components(embeddings: np.ndarray, kernel_matrix: np.ndarray) -> list:
    """
    Compute the number of connected components of graph of affinity matrix / kernel matrix.

    Parameters:
      - dataset: An object with an attribute 'original_data' (the high-dimensional data).
      - embeddings: A numpy array representing the low-dimensional embeddings.
      - kernel_matrix: The kernel matrix of dimensionality reduction algorithm.

    Returns:
      - A list of each connected components .
    """
    graph = nx.from_numpy_array(kernel_matrix)  # Convert adjacency matrix to a graph
    component_sizes = np.sort(np.array([len(k) for k in nx.connected_components(graph)]))[::-1]

    return component_sizes

##############################################################################
# Single-Value Wrappers (conform to Metric(Protocol))
##############################################################################

def ConnectedComponents(dataset, embeddings: np.ndarray, module: LatentModule) -> np.ndarray:
    kernel_matrix = getattr(module, "kernel_matrix", None)
    if kernel_matrix is None:
        warnings.warn(
            "ConnectedComponents metric skipped: module has no 'kernel_matrix' attribute.",
            RuntimeWarning
        )
        return np.array([np.nan])

    return connected_components(embeddings=embeddings, kernel_matrix=kernel_matrix)
