"""Visualize representation trajectories using PHATE."""
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist

from manylatents.algorithms.latent.phate import PHATEModule


@dataclass
class TrajectoryVisualizer:
    """Embed diffusion operator trajectories for visualization.

    Takes a sequence of (step, operator) pairs and embeds them
    in low-dimensional space using PHATE on pairwise distances.

    Distance metrics:
    - frobenius: ||P_i - P_j||_F
    - spectral: Distance between eigenvalue spectra

    Attributes:
        n_components: Output embedding dimension
        distance_metric: How to measure operator distance
        phate_knn: k for PHATE k-NN graph
        phate_t: Diffusion time for PHATE
    """
    n_components: int = 2
    distance_metric: Literal["frobenius", "spectral"] = "frobenius"
    phate_knn: int = 5
    phate_t: int = 10

    def compute_distances(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """Compute pairwise distances between operators in trajectory."""
        operators = [op for _, op in trajectory]

        if self.distance_metric == "frobenius":
            # Flatten and use pdist
            flat = [op.flatten() for op in operators]
            return squareform(pdist(flat, metric="euclidean"))

        elif self.distance_metric == "spectral":
            # Compare eigenvalue spectra
            spectra = []
            for op in operators:
                # Get eigenvalues (sorted descending by magnitude)
                eigvals = np.linalg.eigvalsh(op)
                eigvals = np.sort(np.abs(eigvals))[::-1]
                spectra.append(eigvals)
            return squareform(pdist(spectra, metric="euclidean"))

        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

    def fit_transform(
        self,
        trajectory: List[Tuple[int, np.ndarray]]
    ) -> np.ndarray:
        """Embed trajectory in low-dimensional space.

        Args:
            trajectory: List of (step, operator) tuples

        Returns:
            Array of shape (n_steps, n_components)
        """
        distances = self.compute_distances(trajectory)

        # Convert distance matrix to similarity for PHATE
        # Use Gaussian kernel
        sigma = np.median(distances[distances > 0])
        if sigma == 0:
            sigma = 1.0  # Fallback for identical operators
        similarities = np.exp(-distances**2 / (2 * sigma**2))

        # Use PHATE on the similarity matrix
        phate = PHATEModule(
            n_components=self.n_components,
            knn=min(self.phate_knn, len(trajectory) - 1),
            t=self.phate_t,
        )

        # PHATE expects features, use similarity matrix rows as features
        sim_tensor = torch.from_numpy(similarities).float()
        phate.fit(sim_tensor)
        embedding = phate.transform(sim_tensor)

        return embedding.numpy() if hasattr(embedding, 'numpy') else np.array(embedding)

    def compute_spread(
        self,
        trajectory: List[Tuple[int, np.ndarray]],
    ) -> float:
        """Compute spread metric (average pairwise distance)."""
        distances = self.compute_distances(trajectory)
        # Upper triangle only (excluding diagonal)
        upper_tri = distances[np.triu_indices(len(trajectory), k=1)]
        return float(np.mean(upper_tri))
