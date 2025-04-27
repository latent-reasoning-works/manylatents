import torch
from torch import Tensor
from typing import Optional, Union

import numpy as np
from openTSNE import affinity, initialization
from openTSNE.tsne import TSNEEmbedding
from openTSNE.affinity import PerplexityBasedNN

from .dimensionality_reduction import DimensionalityReductionModule


def build_dense_distance_matrix(distances, neighbors) -> np.ndarray:
    """
    Construct a full NxN matrix from distances and neighbors.
    
    Args:
        distances: NxK array of distances to neighbors
        neighbors:  NxK indices of neighbors

    Returns:
        NxN NumPy array with distances filled in, zeros elsewhere.
    """

    N = neighbors.shape[0]
    matrix = np.zeros((N, N), dtype=distances.dtype)

    for i in range(N):
        matrix[i, neighbors[i]] = distances[i]

    return matrix

class TSNEModule(DimensionalityReductionModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: Optional[float] = 30.0,
        n_iter_early: Optional[int] = 250,
        n_iter_late: Optional[int] = 750,
        learning_rate: Union[float, str] = 'auto',
        metric: str = "euclidean",
        fit_fraction: float = 1.0,
    ):
        super().__init__(n_components, random_state)
        self.perplexity = perplexity
        self.n_iter_early = n_iter_early
        self.n_iter_late = n_iter_late
        self.learning_rate = learning_rate
        self.metric = metric
        self.fit_fraction = fit_fraction
        self.random_state = random_state

    def fit(self, x: Tensor) -> None:
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))
        x_fit = x_np[:n_fit]

        # Step 1: Compute affinities (P matrix)
        
        # monkey patch to allow for large perplexity
        # Overwriting:
        # https://github.com/pavlin-policar/openTSNE/blob/52ae1d67cbe2b99995e6c8dc0fcc3992344998bc/openTSNE/affinity.py#L340
        def do_nothing_check_perplexity(perplexity, k_neighbors):
            # Always just return the perplexity passed in, no checks or clamping
            return perplexity
        PerplexityBasedNN.check_perplexity = staticmethod(do_nothing_check_perplexity)

        self.affinities = PerplexityBasedNN(
            x_fit,
            perplexity=self.perplexity,
            metric=self.metric,
            n_jobs=-1,
            method="approx",  # or "exact"
            random_state=self.random_state
        )

        # Step 2: Initialize embedding
        init = initialization.pca(x_fit, random_state=self.random_state)

        # Step 3: Create Embedding object
        self.embedding_train = TSNEEmbedding(
            init, self.affinities, random_state=self.random_state
        )

        # Step 4: Optimize (i.e., fit)
        self.embedding_train.optimize(
            n_iter=self.n_iter_early,
            learning_rate=self.learning_rate,
            exaggeration=12,  # default in openTSNE
            momentum=0.5,
            inplace=True
        )

        self.embedding_train.optimize(
            n_iter=self.n_iter_late,
            learning_rate=self.learning_rate,
            momentum=0.8,
            inplace=True
        )

        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("tSNE model is not fitted yet. Call `fit` first.")

        x_np = x.detach().cpu().numpy()
        embedding_out = self.embedding_train.transform(x_np)
        return torch.tensor(embedding_out, device=x.device, dtype=x.dtype)

    @property
    def affinity_matrix(self):
        """Returns P matrix"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        P = np.asarray(self.affinities.P.todense())
        return P

    @property
    def kernel_matrix(self):
        """Returns kernel matrix used to build P matrix (not including diagonal)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        # NOTE: affinity matrix has more non-zero entries than kernel matrix.
        # Not sure why.
        # NOTE: values here are >> 1
        #K_no_diag = build_dense_distance_matrix(self.affinities._PerplexityBasedNN__distances,
        #                                        self.affinities._PerplexityBasedNN__neighbors)
        # symmetrize
        #K_no_diag = (K_no_diag + K_no_diag.T) / 2
        K_no_diag = np.asarray(self.affinities.P.todense())

        # add diagonal (just setting to 1)
        #K = np.eye(len(K_no_diag)) + K_no_diag
        K = K_no_diag

        return K