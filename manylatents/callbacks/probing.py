# manylatents/callbacks/probing.py
"""Representation probing utilities and callbacks.

Probes are hook-based observers that extract and analyze representations
during training or after inference. They don't transform the main data flow -
they compute derived quantities (diffusion operators, SAE features, etc.)
that can be logged, visualized, or used for analysis.

Usage:
    # Direct computation
    from manylatents.callbacks.probing import probe
    diff_op = probe(embeddings, method="diffusion")

    # As Lightning callback (see lightning/callbacks/probing.py)
    # As embedding callback (planned)
"""
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from torch import Tensor

from manylatents.utils.kernel_utils import symmetric_diffusion_operator


# =============================================================================
# Core probe dispatch
# =============================================================================

@functools.singledispatch
def probe(
    source: Any,
    /,
    method: str = "diffusion",
    **kwargs,
) -> np.ndarray:
    """Compute probe from representations.

    Args:
        source: Input representations (N samples x D features)
        method: Probe method ("diffusion", future: "sae", "attention")
        **kwargs: Method-specific parameters

    Returns:
        Probe output (shape depends on method)
    """
    raise NotImplementedError(
        f"probe() not implemented for type {type(source)}. "
        f"Expected np.ndarray or torch.Tensor."
    )


@probe.register(np.ndarray)
def _probe_ndarray(source: np.ndarray, /, method: str = "diffusion", **kwargs) -> np.ndarray:
    if method == "diffusion":
        gauge = DiffusionGauge(**kwargs)
        return gauge(source)
    else:
        raise ValueError(f"Unknown probe method: {method}")


@probe.register(Tensor)
def _probe_tensor(source: Tensor, /, method: str = "diffusion", **kwargs) -> np.ndarray:
    if method == "diffusion":
        gauge = DiffusionGauge(**kwargs)
        return gauge(source)
    else:
        raise ValueError(f"Unknown probe method: {method}")


# =============================================================================
# Diffusion probe
# =============================================================================

@dataclass
class DiffusionGauge:
    """Compute diffusion operator from representations.

    Pipeline:
        representations (N, D) -> pairwise distances -> Gaussian kernel ->
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

    def __call__(self, representations: Any) -> np.ndarray:
        """Compute diffusion operator.

        Args:
            representations: Array/Tensor of shape (N, D)

        Returns:
            Diffusion operator of shape (N, N)
        """
        if isinstance(representations, Tensor):
            representations = representations.detach().cpu().numpy()

        distances = squareform(pdist(representations, metric=self.metric))

        if self.knn is not None:
            sorted_dists = np.sort(distances, axis=1)
            sigma = sorted_dists[:, min(self.knn, distances.shape[0] - 1)]
            sigma = np.maximum(sigma, 1e-10)
            kernel = np.exp(-distances**2 / (sigma[:, None] * sigma[None, :]))
        else:
            sigma = np.median(distances[distances > 0])
            kernel = np.exp(-distances**2 / (2 * sigma**2))

        np.fill_diagonal(kernel, 0)

        if self.symmetric:
            return symmetric_diffusion_operator(kernel, alpha=self.alpha)
        else:
            row_sums = kernel.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            return kernel / row_sums


# =============================================================================
# Trajectory analysis (for analyzing probe outputs over time/models)
# =============================================================================

@dataclass
class TrajectoryVisualizer:
    """Embed probe trajectories for visualization.

    Takes a sequence of (step, operator) pairs and embeds them
    in low-dimensional space using PHATE on pairwise distances.
    """
    n_components: int = 2
    distance_metric: Literal["frobenius", "spectral"] = "frobenius"
    phate_knn: int = 5
    phate_t: int = 10

    def compute_distances(self, trajectory: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Compute pairwise distances between operators in trajectory."""
        operators = [op for _, op in trajectory]

        if self.distance_metric == "frobenius":
            flat = [op.flatten() for op in operators]
            return squareform(pdist(flat, metric="euclidean"))
        elif self.distance_metric == "spectral":
            spectra = []
            for op in operators:
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(np.abs(eigvals))[::-1]
                spectra.append(eigvals)
            return squareform(pdist(spectra, metric="euclidean"))
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def fit_transform(self, trajectory: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Embed trajectory in low-dimensional space."""
        from manylatents.algorithms.latent.phate import PHATEModule

        distances = self.compute_distances(trajectory)
        sigma = np.median(distances[distances > 0])
        if sigma == 0:
            sigma = 1.0
        similarities = np.exp(-distances**2 / (2 * sigma**2))

        phate = PHATEModule(
            n_components=self.n_components,
            knn=min(self.phate_knn, len(trajectory) - 1),
            t=self.phate_t,
        )
        sim_tensor = torch.from_numpy(similarities).float()
        phate.fit(sim_tensor)
        embedding = phate.transform(sim_tensor)

        return embedding.numpy() if hasattr(embedding, 'numpy') else np.array(embedding)

    def compute_spread(self, trajectory: List[Tuple[int, np.ndarray]]) -> float:
        """Compute spread metric (average pairwise distance)."""
        distances = self.compute_distances(trajectory)
        upper_tri = distances[np.triu_indices(len(trajectory), k=1)]
        return float(np.mean(upper_tri))


def compute_multi_model_spread(
    trajectories: List[List[Tuple[int, np.ndarray]]],
    distance_metric: str = "frobenius",
) -> np.ndarray:
    """Compute spread across models at each timestep.

    Lower spread indicates models are converging to similar representations.
    """
    n_steps = len(trajectories[0])
    n_models = len(trajectories)
    spreads = []

    for step_idx in range(n_steps):
        operators = [traj[step_idx][1] for traj in trajectories]

        if distance_metric == "frobenius":
            flat = [op.flatten() for op in operators]
            if n_models > 1:
                dists = pdist(flat, metric="euclidean")
                spread = float(np.mean(dists))
            else:
                spread = 0.0
        elif distance_metric == "spectral":
            spectra = []
            for op in operators:
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(np.abs(eigvals))[::-1]
                spectra.append(eigvals)
            if n_models > 1:
                dists = pdist(spectra, metric="euclidean")
                spread = float(np.mean(dists))
            else:
                spread = 0.0
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        spreads.append(spread)

    return np.array(spreads)
