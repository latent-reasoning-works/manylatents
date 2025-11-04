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
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union
from deprecated import deprecated

from .latent_module_base import LatentModule
import logging
logger = logging.getLogger(__name__)
import scprep
import s_gd2


class MultidimensionalScaling():
    def __init__(self, 
                 ndim: int = 2,
                seed: Optional[int] = 42,
                how: str = "metric", #  choose from ['classic', 'metric', 'nonmetric']
                solver: str = "sgd", # choose from ["sgd", "smacof"]
                distance_metric: str = 'euclidean', # recommended values: 'euclidean' and 'cosine'
                n_jobs: Optional[int] = -1,
                verbose = False,
                ):
        self.ndim = ndim
        self.seed=seed
        self.how = how
        self.solver = solver
        self.distance_metric = distance_metric
        self.n_jobs=n_jobs
        self.verbose=verbose

        self.embedding = None

    # Fast classical MDS using random svd
    @deprecated(version="1.0.0", reason="Use phate.mds.classic instead")
    def cmdscale_fast(self, D):
        return self.classic(D=D)


    def classic(self, D):
        """Fast CMDS using random SVD

        Parameters
        ----------
        D : array-like, shape=[n_samples, n_samples]
            pairwise distances

        n_components : int, optional (default: 2)
            number of dimensions in which to embed `D`

        random_state : int, RandomState or None, optional (default: None)
            numpy random state

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


    @scprep.utils._with_pkg(pkg="s_gd2", min_version="1.3")
    def sgd(self, D, init=None):
        """Metric MDS using stochastic gradient descent

        Parameters
        ----------
        D : array-like, shape=[n_samples, n_samples]
            pairwise distances

        n_components : int, optional (default: 2)
            number of dimensions in which to embed `D`

        random_state : int or None, optional (default: None)
            numpy random state

        init : array-like or None
            Initialization algorithm or state to use for MMDS

        Returns
        -------
        Y : array-like, embedded data [n_sample, ndim]
        """
        if not self.ndim == 2:
            raise NotImplementedError
        N = D.shape[0]
        D = squareform(D)
        # Metric MDS from s_gd2
        Y = s_gd2.mds_direct(N, D, init=init, random_seed=self.seed)
        return Y


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

        n_components : int, optional (default: 2)
            number of dimensions in which to embed `D`

        metric : bool, optional (default: True)
            Use metric MDS. If False, uses non-metric MDS

        init : array-like or None, optional (default: None)
            Initialization state

        random_state : int, RandomState or None, optional (default: None)
            numpy random state

        verbose : int or bool, optional (default: 0)
            verbosity

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


    def embed_MDS(self, X):
        """Performs classic, metric, and non-metric MDS

        Metric MDS is initialized using classic MDS,
        non-metric MDS is initialized using metric MDS.

        Parameters
        ----------
        X: ndarray [n_samples, n_features]
            2 dimensional input data array with n_samples

        n_dim : int, optional, default: 2
            number of dimensions in which the data will be embedded

        how : string, optional, default: 'classic'
            choose from ['classic', 'metric', 'nonmetric']
            which MDS algorithm is used for dimensionality reduction

        distance_metric : string, optional, default: 'euclidean'
            choose from ['cosine', 'euclidean']
            distance metric for MDS

        solver : {'sgd', 'smacof'}, optional (default: 'sgd')
            which solver to use for metric MDS. SGD is substantially faster,
            but produces slightly less optimal results. Note that SMACOF was used
            for all figures in the PHATE paper.

        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used

        seed: integer or numpy.RandomState, optional
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global numpy random number generator

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

        # MDS embeddings, each gives a different output.
        X_dist = squareform(pdist(X, self.distance_metric))

        # initialize all by CMDS
        Y_classic = self.classic(X_dist)
        if self.how == "classic":
            self.embedding = Y_classic
            return Y_classic

        # metric is next fastest
        if self.solver == "sgd":
            try:
                # use sgd2 if it is available
                Y = self.sgd(X_dist, init=Y_classic)
                if np.any(~np.isfinite(Y)):
                    logger.warning("Using SMACOF because SGD returned NaN")
                    raise NotImplementedError
            except NotImplementedError:
                # sgd2 currently only supports n_components==2
                Y = self.smacof(
                    X_dist,
                    init=Y_classic,
                    metric=True,
                )
        elif self.solver == "smacof":
            Y = self.smacof(
                X_dist, init=Y_classic, metric=True
            )
        else:
            raise RuntimeError
        if self.how == "metric":
            # re-orient to classic
            _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
            self.embedding = Y
            return Y

        # nonmetric is slowest
        Y = self.smacof(X_dist, init=Y, metric=False)
        # re-orient to classic
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
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
                                            verbose=verbose)

    def fit(self, x: Tensor) -> None:
        """Fits MDS on all of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        emb = self.model.embed_MDS(x_np[:n_fit])
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
        embedding = self.model.embed_MDS(x_np)
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

        The eigenvalues of this normalized Gram matrix represent the variance
        structure that MDS preserves, analogous to PCA's variance spectrum.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.
            use_symmetric: Ignored for MDS (always symmetric). Default False.

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