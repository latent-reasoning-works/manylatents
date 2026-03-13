"""
Multidimensional Scaling (MDS) dimensionality reduction.

Original author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
(C) 2017 Krishnaswamy Lab GPLv2
Revised by: Shuang Ni 2025

Note: This file contains both the core algorithm implementation (MultidimensionalScaling class)
and the PyTorch Lightning wrapper (MDSModule). Ideally, these should be separate with the
core implementation imported from a shared library, but they are combined here for convenience.
"""

from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.spatial
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union

from .latent_module_base import LatentModule
import logging
import warnings
logger = logging.getLogger(__name__)


class MultidimensionalScaling():
    def __init__(self,
                 ndim: int = 2,
                seed: Optional[int] = 42,
                how: str = "metric", #  choose from ['classic', 'metric', 'nonmetric']
                solver: str = "sgd", # choose from ["sgd", "smacof"]
                distance_metric: str = 'euclidean', # recommended values: 'euclidean' and 'cosine'
                n_jobs: Optional[int] = -1,
                verbose = False,
                n_landmark: Optional[int] = None,
                random_landmarking: bool = False,
                ):
        self.ndim = ndim
        self.seed=seed
        self.how = how
        self.solver = solver
        self.distance_metric = distance_metric
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.n_landmark = n_landmark
        self.random_landmarking = random_landmarking

        self.embedding = None
        self.distance_matrix = None  # Store computed distance matrix
        self._landmark_indices = None

    def classic(self, D):
        """Fast CMDS using random SVD

        Parameters
        ----------
        D : array-like, shape=[n_samples, n_samples]
            pairwise distances

        Returns
        -------
        Y : array-like, embedded data [n_sample, ndim]
        """
        D = D**2
        D = D - D.mean(axis=0)[None, :]
        D = D - D.mean(axis=1)[:, None]
        pca = PCA(
            n_components=self.ndim, svd_solver="randomized", random_state=self.seed
        )
        Y = pca.fit_transform(D)
        return Y

    def sgd(self, D, init=None):
        """Metric MDS using stochastic gradient descent via phate.sgd_mds.

        Parameters
        ----------
        D : array-like, shape=[n_samples, n_samples]
            pairwise distances
        init : array-like or None
            Initial embedding

        Returns
        -------
        Y : array-like, shape=[n_samples, n_components]
        """
        from phate.sgd_mds import sgd_mds
        return sgd_mds(
            D,
            n_components=self.ndim,
            init=init,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def smacof(
            self,
            D,
            metric=True,
            init=None,
            max_iter=3000,
            eps=1e-6,
        ):
        """Metric and non-metric MDS using SMACOF

        Parameters
        ----------
        D : array-like, shape=[n_samples, n_samples]
            pairwise distances
        metric : bool, optional (default: True)
            Use metric MDS. If False, uses non-metric MDS
        init : array-like or None, optional (default: None)
            Initialization state
        max_iter : int, optional (default: 3000)
            maximum iterations
        eps : float, optional (default: 1e-6)
            stopping criterion

        Returns
        -------
        Y : array-like, shape=[n_samples, n_components]
            embedded data
        """
        # Metric MDS from sklearn
        Y, _ = manifold.smacof(
            D,
            n_components=self.ndim,
            metric=metric,
            max_iter=max_iter,
            eps=eps,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            n_init=1,
            init=init,
            verbose=self.verbose,
        )
        return Y

    def _select_landmarks(self, n_samples):
        """Select landmark indices.

        Parameters
        ----------
        n_samples : int
            Total number of samples

        Returns
        -------
        landmark_indices : ndarray of shape (n_landmark,)
        """
        if self.random_landmarking:
            rng = np.random.RandomState(self.seed)
            indices = rng.choice(n_samples, self.n_landmark, replace=False)
            indices.sort()
            return indices
        else:
            return np.linspace(0, n_samples - 1, self.n_landmark, dtype=int)

    def _extend_to_nonlandmarks(self, X, landmark_indices, landmark_embedding, k=8):
        """Extend landmark embedding to all points via inverse-distance-weighted interpolation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Full dataset
        landmark_indices : ndarray of shape (n_landmark,)
            Indices of landmarks in X
        landmark_embedding : ndarray, shape (n_landmark, n_components)
            Embedding of landmark points
        k : int
            Number of nearest landmarks to use for interpolation

        Returns
        -------
        full_embedding : ndarray, shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        full_embedding = np.empty((n_samples, landmark_embedding.shape[1]))

        # Place landmark embeddings
        full_embedding[landmark_indices] = landmark_embedding

        # Find non-landmark indices
        all_indices = np.arange(n_samples)
        mask = np.ones(n_samples, dtype=bool)
        mask[landmark_indices] = False
        nonlandmark_indices = all_indices[mask]

        if len(nonlandmark_indices) == 0:
            return full_embedding

        # Compute distances from non-landmarks to landmarks
        X_landmarks = X[landmark_indices]
        X_nonlandmarks = X[nonlandmark_indices]
        D_nl = cdist(X_nonlandmarks, X_landmarks, metric=self.distance_metric)

        # Find k nearest landmarks using argpartition
        k_actual = min(k, len(landmark_indices))
        knn_indices = np.argpartition(D_nl, k_actual, axis=1)[:, :k_actual]

        # Gather distances to k nearest landmarks
        rows = np.arange(len(nonlandmark_indices))[:, None]
        knn_dists = D_nl[rows, knn_indices]

        # Inverse-distance weights (add small epsilon to avoid division by zero)
        weights = 1.0 / (knn_dists + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)

        # Weighted average of landmark embeddings
        knn_embeddings = landmark_embedding[knn_indices]  # (n_nonlandmarks, k, n_components)
        full_embedding[nonlandmark_indices] = np.einsum('ij,ijk->ik', weights, knn_embeddings)

        return full_embedding

    def embed_MDS(self, X):
        """Performs classic, metric, and non-metric MDS

        Metric MDS is initialized using classic MDS,
        non-metric MDS is initialized using metric MDS.

        Parameters
        ----------
        X: ndarray [n_samples, n_features]
            2 dimensional input data array with n_samples

        Returns
        -------
        Y : ndarray [n_samples, n_dim]
            low dimensional embedding of X using MDS
        """

        if self.how not in ["classic", "metric", "nonmetric"]:
            raise ValueError(
                "Allowable 'how' values for MDS: 'classic', "
                "'metric', or 'nonmetric'. "
                "'{}' was passed.".format(self.how)
            )
        if self.solver not in ["sgd", "smacof"]:
            raise ValueError(
                "Allowable 'solver' values for MDS: 'sgd' or "
                "'smacof'. "
                "'{}' was passed.".format(self.solver)
            )

        n_samples = X.shape[0]
        use_landmarks = (self.n_landmark is not None and self.n_landmark < n_samples)

        if use_landmarks:
            self._landmark_indices = self._select_landmarks(n_samples)
            X_landmarks = X[self._landmark_indices]
            self.distance_matrix = squareform(pdist(X_landmarks, self.distance_metric))
            D = self.distance_matrix
        else:
            self._landmark_indices = None
            self.distance_matrix = squareform(pdist(X, self.distance_metric))
            D = self.distance_matrix

        # initialize all by CMDS
        Y_classic = self.classic(D)
        if self.how == "classic":
            if use_landmarks:
                Y_full = self._extend_to_nonlandmarks(X, self._landmark_indices, Y_classic)
                self.embedding = Y_full
                return Y_full
            self.embedding = Y_classic
            return Y_classic

        # metric is next fastest
        if self.solver == "sgd":
            Y = self.sgd(D, init=Y_classic)
        elif self.solver == "smacof":
            Y = self.smacof(D, init=Y_classic, metric=True)
        else:
            raise RuntimeError

        if self.how == "metric":
            # re-orient to classic
            _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
            if use_landmarks:
                Y_full = self._extend_to_nonlandmarks(X, self._landmark_indices, Y)
                self.embedding = Y_full
                return Y_full
            self.embedding = Y
            return Y

        # nonmetric is slowest
        Y = self.smacof(D, init=Y, metric=False)
        # re-orient to classic
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
        if use_landmarks:
            Y_full = self._extend_to_nonlandmarks(X, self._landmark_indices, Y)
            self.embedding = Y_full
            return Y_full
        self.embedding = Y
        return Y


# PyTorch Lightning Module wrapper

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
        n_landmark: Optional[int] = None,
        random_landmarking: bool = False,
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.model = MultidimensionalScaling(ndim=n_components,
                                            seed=random_state,
                                            how=how,
                                            solver=solver,
                                            distance_metric=distance_metric,
                                            n_jobs=n_jobs,
                                            verbose=verbose,
                                            n_landmark=n_landmark,
                                            random_landmarking=random_landmarking)

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Fits MDS on all of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data

        # embed_MDS will compute and store distance matrix in self.model.distance_matrix
        emb = self.model.embed_MDS(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted MDS model. MDS can't be extend to new data, so we just return the embedding of the fitted data."""
        if not self._is_fitted:
            raise RuntimeError("MDS model is not fitted yet. Call `fit` first.")

        embedding = self.model.embedding
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def fit_transform(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Fit and then transform on same data."""
        x_np = x.detach().cpu().numpy()

        # embed_MDS will compute and store distance matrix in self.model.distance_matrix
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

    def affinity_matrix(self, ignore_diagonal: bool = False, use_symmetric: bool = False) -> np.ndarray:
        """
        Returns normalized Gram matrix (double-centered squared distance matrix / (n-1)).

        For MDS, the appropriate "affinity" is the Gram matrix that classical MDS
        uses internally, normalized by (n-1) so eigenvalues represent variance.
        This is computed as G = -0.5 * H * D^2 * H' / (n-1) where H is
        the centering matrix and D is the distance matrix.

        When landmarks are used, returns the n_landmark x n_landmark Gram matrix
        (the structure MDS operated on).

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
            use_symmetric: Ignored for MDS (always symmetric). Default False.

        Returns:
            N×N normalized Gram matrix (eigenvalues = variance explained).
        """
        if not self._is_fitted:
            raise RuntimeError("MDS model is not fitted yet. Call `fit` first.")

        if self.model.distance_matrix is None:
            raise RuntimeError("Distance matrix not available. This should not happen.")

        # Compute Gram matrix following classical MDS procedure
        D_squared = self.model.distance_matrix ** 2

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
