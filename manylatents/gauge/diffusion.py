"""Compute diffusion operators from activation tensors."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from torch import Tensor

from manylatents.utils.kernel_utils import symmetric_diffusion_operator


@dataclass
class DiffusionGauge:
    """Compute diffusion operator from activation tensors.

    Pipeline:
        activations (N, D) -> pairwise distances -> Gaussian kernel ->
        affinity matrix -> diffusion operator

    Attributes:
        knn: Number of neighbors for adaptive bandwidth. If None, uses global bandwidth.
        alpha: Diffusion normalization parameter (0=graph Laplacian, 1=Laplace-Beltrami)
        symmetric: If True, return symmetric operator D^{-1/2} K D^{-1/2}
        metric: Distance metric for pairwise computation
    """
    knn: Optional[int] = 15
    alpha: float = 1.0
    symmetric: bool = False
    metric: str = "euclidean"

    def __call__(self, activations: Tensor) -> np.ndarray:
        """Compute diffusion operator from activations.

        Args:
            activations: Tensor of shape (N, D) - N samples, D dimensions

        Returns:
            Diffusion operator of shape (N, N)
        """
        if isinstance(activations, Tensor):
            activations = activations.detach().cpu().numpy()

        # Compute pairwise distances
        distances = squareform(pdist(activations, metric=self.metric))

        # Compute adaptive bandwidth (local scaling)
        if self.knn is not None:
            # k-th nearest neighbor distance for each point
            sorted_dists = np.sort(distances, axis=1)
            sigma = sorted_dists[:, min(self.knn, distances.shape[0] - 1)]
            sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero
            # Adaptive Gaussian kernel
            kernel = np.exp(-distances**2 / (sigma[:, None] * sigma[None, :]))
        else:
            # Global bandwidth (median heuristic)
            sigma = np.median(distances[distances > 0])
            kernel = np.exp(-distances**2 / (2 * sigma**2))

        # Zero out diagonal for cleaner diffusion
        np.fill_diagonal(kernel, 0)

        if self.symmetric:
            return symmetric_diffusion_operator(kernel, alpha=self.alpha)
        else:
            # Row-stochastic normalization
            row_sums = kernel.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            return kernel / row_sums
