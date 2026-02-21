"""
Multiscale PHATE dimensionality reduction with diffusion condensation.

This module wraps the multiscale_phate library to provide hierarchical
embedding with stable component detection across scales.
"""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from .latent_module_base import LatentModule


class MultiscalePHATEModule(LatentModule):
    """
    Multiscale PHATE for hierarchical dimensionality reduction.

    Uses diffusion condensation to identify stable resolutions and compute
    embeddings at multiple scales. Useful for detecting the number of
    connected components (β₀) that are stable across coarsening.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the final embedding.
    scale : float, default=1.025
        Epsilon growth rate for condensation. Higher = faster coarsening.
    granularity : float, default=0.1
        Merge threshold sensitivity. Lower = more aggressive merging.
    landmarks : int, default=2000
        Number of landmarks for scalability.
    knn : int, default=5
        Number of nearest neighbors for graph construction.
    decay : int, default=40
        Kernel tail decay rate.
    gamma : float, default=1
        Informational distance constant.
    n_pca : int, optional
        Number of PCA components for preprocessing. None = auto.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        scale: float = 1.025,
        granularity: float = 0.1,
        landmarks: int = 2000,
        knn: int = 5,
        decay: int = 40,
        gamma: float = 1.0,
        n_pca: Optional[int] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = 42,
        neighborhood_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(n_components=n_components, init_seed=random_state,
                         neighborhood_size=neighborhood_size, **kwargs)

        self.scale = scale
        self.granularity = granularity
        self.landmarks = landmarks
        self.knn = neighborhood_size if neighborhood_size is not None else knn
        self.decay = decay
        self.gamma = gamma
        self.n_pca = n_pca
        self.n_jobs = n_jobs
        self.random_state = random_state

        from multiscale_phate import Multiscale_PHATE

        self.model = Multiscale_PHATE(
            scale=scale,
            granularity=granularity,
            landmarks=landmarks,
            knn=self.knn,
            decay=decay,
            gamma=gamma,
            n_pca=n_pca,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # Store results after fitting
        self.embedding_: Optional[np.ndarray] = None
        self.clusters_: Optional[np.ndarray] = None
        self.sizes_: Optional[np.ndarray] = None

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """
        Fit Multiscale PHATE and compute embedding at finest resolution.

        Parameters
        ----------
        x : Tensor
            Input data of shape (n_samples, n_features).
        """
        x_np = x.detach().cpu().numpy()
        self._n_input = x_np.shape[0]
        self.embedding_, self.clusters_, self.sizes_ = self.model.fit_transform(x_np)
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """
        Return the embedding at finest resolution.

        Multiscale PHATE computes the embedding during fit. Condensation may
        merge nearby points, producing fewer rows than the input. When that
        happens, the embedding is padded with zeros so the output shape matches
        the input (required by downstream callbacks like PlotEmbeddings).

        Parameters
        ----------
        x : Tensor
            Input data (used for device/dtype and expected row count).

        Returns
        -------
        Tensor
            2D embedding of shape (n_samples, n_components).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Multiscale PHATE model is not fitted yet. Call `fit` first."
            )
        n_expected = x.shape[0]
        n_emb = self.embedding_.shape[0]
        if n_emb < n_expected:
            # Pad: assign condensed-away points the mean embedding
            padded = np.zeros((n_expected, self.embedding_.shape[1]))
            padded[:n_emb] = self.embedding_
            padded[n_emb:] = self.embedding_.mean(axis=0)
            return torch.tensor(padded, device=x.device, dtype=x.dtype)
        return torch.tensor(self.embedding_, device=x.device, dtype=x.dtype)

    def get_n_components_per_scale(self) -> np.ndarray:
        """
        Get number of connected components (β₀) at each condensation scale.

        Returns
        -------
        np.ndarray
            Array of component counts, one per scale. Monotonically non-increasing.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        return np.array([len(np.unique(c)) for c in self.model.NxTs])

    def get_gradient(self) -> np.ndarray:
        """
        Get gradient profile across scales.

        Low gradient indicates stable resolutions where component count
        doesn't change rapidly.

        Returns
        -------
        np.ndarray
            Gradient values at each scale transition.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        return self.model.gradient

    def get_stable_scales(self) -> List[int]:
        """
        Get indices of stable (salient) resolutions.

        These are scales where the gradient is low, indicating stable
        component structure.

        Returns
        -------
        List[int]
            Indices into the scale array for stable resolutions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        return list(self.model.levels)

    def get_cluster_assignments(self, scale_idx: int = 0) -> np.ndarray:
        """
        Get cluster assignments at a specific scale.

        Parameters
        ----------
        scale_idx : int, default=0
            Index into NxTs. 0 = finest (most clusters), -1 = coarsest.

        Returns
        -------
        np.ndarray
            Cluster labels for each original data point.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        return self.model.NxTs[scale_idx]

    def get_n_stable_components(self) -> int:
        """
        Get number of components at the most stable resolution.

        Convenience method that returns β₀ at the first stable scale.

        Returns
        -------
        int
            Number of stable connected components.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        stable_scales = self.get_stable_scales()
        if not stable_scales:
            # Fallback to coarsest scale
            return len(np.unique(self.model.NxTs[-1]))
        return len(np.unique(self.model.NxTs[stable_scales[0]]))
