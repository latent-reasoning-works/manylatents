import torch
from torch import Tensor
from typing import Optional, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ..mds_algorithm import MultidimensionalScaling
from ..latent_module_base import LatentModule


class MDSModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        how: str = "metric", #  choose from ['classic', 'metric', 'nonmetric']
        solver: str = "smacof", # choose from ["sgd", "smacof"]
        distance_metric: str = 'euclidean', # recommended values: 'euclidean' and 'cosine'
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.distance_metric = distance_metric
        self.model = MultidimensionalScaling(ndim=n_components,
                                            seed=random_state,
                                            how=how,
                                            solver=solver,
                                            distance_metric=distance_metric,
                                            n_jobs=n_jobs,
                                            verbose=verbose)
        self._distance_matrix = None  # Store distance matrix

    def fit(self, x: Tensor) -> None:
        """Fits MDS on all of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        x_fit = x_np[:n_fit]

        # Compute and store distance matrix
        self._distance_matrix = squareform(pdist(x_fit, self.distance_metric))

        # Fit MDS
        emb = self.model.embed_MDS(x_fit)
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted MDS model. MDS can't be extend to new data, so we just return the embedding of the fitted data."""
        if not self._is_fitted:
            raise RuntimeError("MDS model is not fitted yet. Call `fit` first.")

        embedding = self.model.embedding
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()

        # Compute and store distance matrix
        self._distance_matrix = squareform(pdist(x_np, self.distance_metric))

        embedding = self.model.embed_MDS(x_np)
        self._is_fitted = True
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns Gram matrix (same as affinity_matrix for MDS).

        MDS doesn't have a meaningful kernel matrix in the same sense as methods
        like UMAP or PHATE. The distance matrix is not appropriate for metrics
        that expect similarity/affinity matrices. Instead, we return the Gram
        matrix, which is what classical MDS actually uses internally.

        For the raw distance matrix, access self._distance_matrix directly.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N normalized Gram matrix (same as affinity_matrix).
        """
        return self.affinity_matrix(ignore_diagonal=ignore_diagonal)

    def affinity_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns normalized Gram matrix (double-centered squared distance matrix / (n-1)).

        For MDS, the appropriate "affinity" is the Gram matrix that classical MDS
        uses internally, normalized by (n-1) so eigenvalues represent variance.
        This is computed as G = -0.5 * H * D^2 * H' / (n-1) where H is
        the centering matrix and D is the distance matrix.

        The eigenvalues of this normalized Gram matrix represent the variance
        structure that MDS preserves, analogous to PCA's variance spectrum.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N normalized Gram matrix (eigenvalues = variance explained).
        """
        if not self._is_fitted:
            raise RuntimeError("MDS model is not fitted yet. Call `fit` first.")

        if self._distance_matrix is None:
            raise RuntimeError("Distance matrix not available. This should not happen.")

        # Compute Gram matrix following classical MDS procedure
        D_squared = self._distance_matrix ** 2

        # Double-center: G = -0.5 * H * D^2 * H where H = I - (1/n)*11'
        n = D_squared.shape[0]
        row_means = D_squared.mean(axis=1, keepdims=True)
        col_means = D_squared.mean(axis=0, keepdims=True)
        grand_mean = D_squared.mean()

        G = -0.5 * (D_squared - row_means - col_means + grand_mean)

        # Normalize by (n-1) to get variance scale
        G = G / (n - 1)

        if ignore_diagonal:
            G = G - np.diag(np.diag(G))

        return G