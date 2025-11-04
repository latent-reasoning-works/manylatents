"""
Diffusion Map dimensionality reduction.

Note: This file contains both the core algorithm implementation (DiffusionMap class)
and the PyTorch Lightning wrapper (DiffusionMapModule). Ideally, these should be 
separate with the core implementation imported from a shared library, but they are
combined here for convenience.
"""

import graphtools
import numpy as np
import scipy
import torch
from torch import Tensor
from typing import Optional, Union
import logging

from .latent_module_base import LatentModule

logger = logging.getLogger(__name__)

def compute_dm(K, alpha=0., verbose=0):
    # Using setup and notation from: https://www.stats.ox.ac.uk/~cucuring/CDT_08_Nonlinear_Dim_Red_b__Diffusion_Maps_FoDS.pdf
    # alpha=0 Graph Laplacian
    # alpha=0.5 Fokker-Plank operator
    # alpha=1 Laplace-Beltrami operator

    #d_noalpha = K.sum(1).flatten()
    d_noalpha = K.sum(axis=1)

    # Check for degenerate kernel (zero or near-zero row sums)
    if np.any(np.abs(d_noalpha) < 1e-10):
        n = K.shape[0]
        evecs_right = np.full((n, n), np.nan)
        evals = np.full(n, np.nan)
        L = np.full((n, n), np.nan)
        return evecs_right, evals, L, d_noalpha

    # weighted graph Laplacian normalization
    d = d_noalpha**alpha
    D = np.diag(d)
    D_inv = np.diag(1/d)

    # Normalize K according to α
    K_alpha = D_inv@K@D_inv

    #d_alpha = K_alpha.sum(1).flatten()
    d_alpha = K_alpha.sum(axis=1)
    D_alpha = np.diag(d_alpha)
    D_inv_alpha = np.diag(1/d_alpha)

    L = D_inv_alpha@K_alpha # anisotropic transition kernel (AKA diffusion operator)

    # build symmetric matrix
    D_sqrt_inv_alpha = np.diag(1/np.sqrt(d_alpha))
    D_sqrt_alpha = np.diag(np.sqrt(d_alpha))
    S = D_sqrt_inv_alpha@K_alpha@D_sqrt_inv_alpha

    # spectral decomposition of S
    # IMPORTANT NOTE:
    # In theory you could run np.linalg.eig(L),
    # BUT this returns non-orthogonal eigenvectors!
    # So would have to correct for that
    # Using SVD since more numerically stable
    evecs, svals, _ = scipy.linalg.svd(S)

    # Retrieve sign!
    test_product = S@evecs
    expected_product = evecs@np.diag(svals)
    signs = np.isclose(expected_product, test_product).mean(0)
    signs[~np.isclose(signs, 1)] = -1
    evals = svals*signs

    # convert right eigenvectors of S to those of L
    evecs_right = D_sqrt_inv_alpha@evecs
    evecs_left = D_sqrt_alpha@evecs

    # make sure ordered by eigenvalue
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    #evecs = evecs[:, order]
    evecs_left = evecs_left[:, order]
    evecs_right = evecs_right[:, order]

    # Scaling factor for eigenvectors
    #scaling_factor = 1/d_noalpha.sum()
    #scaling_factor = evecs_right[0, 0]
    scaling_factor = 1/np.sqrt(d_alpha.sum())
    if np.isclose(evecs_right[:,0]/scaling_factor, -1).any():
        scaling_factor *= -1

    # Apply scaling
    evecs_right /= scaling_factor

    # Adjust left eigenvectors to maintain eigendecomposition
    # Assuming evecs_right[0,0] is non-zero for all practical cases
    evecs_left *= scaling_factor

    # Safety Checks!
    # First left eigenvector is stationary distribution
    if verbose > 0:
        neg_evals = evals < 0
        if neg_evals.sum() > 0:
            print("{} eigenvalues are negative: min={}".format(len(evals[neg_evals]),
                                                               evals[neg_evals].min()))
        one_evals = np.isclose(evals, 1).sum()
        if one_evals > 1:
            print("{} eigenvalues are 1".format(one_evals))
        if not np.allclose(evecs_left[:,0]/evecs_left.sum(), d_alpha/d_alpha.sum()):
            print("left evec not exactly stationary dist. Proceed with caution!")
        # First right eigenvector is all 1s
        if not  np.allclose(evecs_right[:,0], 1):
            print("right evec not trivial (1s)! Proceed with caution!")
        # Decomposition is correct
        if not np.allclose(L, evecs_right@np.diag(evals)@evecs_left.T):
            print("evals/evecs dont exactly match with diffusion operator. Proceed with caution!")

    #diffusion_coords = evecs_right@np.diag(evals)
    
    # return eigenvectors and eigenvalues instead so that we can compute DMs for different t's
    return evecs_right, evals, L, d_noalpha


def matrix_is_equivalent(ref, new):
    return ref.shape[1] == new.shape[1]


class DiffusionMap():
    """Implementation of Diffusion Maps

    Parameters
    ----------

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    t : int, optional, default: 'auto'
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator

    knn : int, optional, default: 5
        number of nearest neighbors on which to build kernel

    decay : int, optional, default: 40
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    n_landmark : int, optional, default: 2000
        number of landmarks to use in fast PHATE

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

    knn_dist : string, optional, default: 'euclidean'
        recommended values: 'euclidean', 'cosine', 'precomputed'
        Any metric from `scipy.spatial.distance` can be used
        distance metric for building kNN graph. Custom distance
        functions of form `f(x, y) = d` are also accepted. If 'precomputed',
        `data` should be an n_samples x n_samples distance or
        affinity matrix. Distance matrices are assumed to have zeros
        down the diagonal, while affinity matrices are assumed to have
        non-zero values down the diagonal. This is detected automatically
        using `data[0,0]`. You can override this detection with
        `knn_dist='precomputed_distance'` or `knn_dist='precomputed_affinity'`.

    knn_max : int, optional, default: None
        Maximum number of neighbors for which alpha decaying kernel
        is computed for each point. For very large datasets, setting `knn_max`
        to a small multiple of `knn` can speed up computation significantly.

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global `numpy` random number generator

    verbose : `int` or `boolean`, optional (default: 1)
        If `True` or `> 0`, print status messages

    Returns
    -------
    self
    """
    def __init__(self, 
                 n_components=2, 
                 t=1, 
                 n_pca=None, 
                 n_landmark=None,
                 knn=5,
                 knn_max=100,
                 knn_dist='euclidean',
                 decay=40,
                 n_jobs=8,
                 verbose=0,
                 random_state=42):
        self.n_components = n_components
        self.n_landmark = n_landmark
        self.t = t
        self.n_pca = n_pca
        self.knn=knn
        self.knn_max=knn_max
        self.knn_dist = knn_dist
        self.decay=decay
        self.n_jobs=n_jobs
        self.verbose=verbose
        self.random_state=random_state
        self.X = None
        self.G = None

    def fit(self, X):

        if self.n_landmark is None or X.shape[0] <= self.n_landmark:
            n_landmark = None
        else:
            n_landmark = self.n_landmark

        if self.G is None:
            self.G = graphtools.Graph(X, 
                                      n_pca=self.n_pca, 
                                      n_landmark=self.n_landmark,
                                      distance=self.knn_dist,
                                      knn=self.knn,
                                      knn_max=self.knn_max,
                                      decay=self.decay,
                                      thresh=1e-4,
                                      n_jobs=self.n_jobs,
                                      verbose=self.verbose,
                                      random_state=self.random_state)

        K = self.G.kernel
        K = np.array(K.todense())
        self.evecs_right, self.evals, _, _ = compute_dm(K, 1)
        self.X = X
        
        
    
    @property
    def embedding(self):
        if not hasattr(self, "evecs_right") or not hasattr(self, "evals"):
            raise ValueError("Model has not been fitted yet. Call `.fit()` first.")
        return self.evecs_right @ np.diag(self.evals**self.t)[:, 1:(self.n_components+1)]

    def transform(self, X=None):
        if self.embedding is None:
            raise Exception("Need to fit model first!")
        if X is None:
            return self.embedding
        # process case where X different from fitted X
        if matrix_is_equivalent(X, self.X):
            transitions = self.G.extend_to_data(X)
            return self.G.interpolate(self.embedding, transitions)
        else:
            raise Exception("Trying to transform data of different dimension from what was fitted!")

    def set_params(self, **params):
        """Set the parameters on this estimator.

        Any parameters not given as named arguments will be left at their
        current value.

        Parameters
        ----------

        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        knn : int, optional, default: 5
            number of nearest neighbors on which to build kernel

        decay : int, optional, default: 40
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        n_landmark : int, optional, default: 2000
            number of landmarks to use in fast PHATE

        t : int, optional, default: 'auto'
            power to which the diffusion operator is powered.
            This sets the level of diffusion. If 'auto', t is selected
            according to the knee point in the Von Neumann Entropy of
            the diffusion operator

        n_pca : int, optional, default: 100
            Number of principal components to use for calculating
            neighborhoods. For extremely large datasets, using
            n_pca < 20 allows neighborhoods to be calculated in
            roughly log(n_samples) time.

        knn_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean', 'cosine', 'precomputed'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for building kNN graph. Custom distance
            functions of form `f(x, y) = d` are also accepted. If 'precomputed',
            `data` should be an n_samples x n_samples distance or
            affinity matrix. Distance matrices are assumed to have zeros
            down the diagonal, while affinity matrices are assumed to have
            non-zero values down the diagonal. This is detected automatically
            using `data[0,0]`. You can override this detection with
            `knn_dist='precomputed_distance'` or `knn_dist='precomputed_affinity'`.

        knn_max : int, optional, default: None
            Maximum number of neighbors for which alpha decaying kernel
            is computed for each point. For very large datasets, setting `knn_max`
            to a small multiple of `knn` can speed up computation significantly.

        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used

        random_state : integer or numpy.RandomState, optional, default: None
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global `numpy` random number generator

        verbose : `int` or `boolean`, optional (default: 1)
            If `True` or `> 0`, print status messages

        Returns
        -------
        self
        """
        reset_kernel = False
        reset_embedding = False

        # embedding parameters
        if "n_components" in params and params["n_components"] != self.n_components:
            self.n_components = params["n_components"]
            del params["n_components"]
            
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            del params["t"]

        # kernel parameters
        if "knn" in params and params["knn"] != self.knn:
            self.knn = params["knn"]
            reset_kernel = True
            del params["knn"]
        if "knn_max" in params and params["knn_max"] != self.knn_max:
            self.knn_max = params["knn_max"]
            reset_kernel = True
            del params["knn_max"]
        if "decay" in params and params["decay"] != self.decay:
            self.decay = params["decay"]
            reset_kernel = True
            del params["decay"]
        if "n_pca" in params:
            if self.X is not None and params["n_pca"] >= np.min(self.X.shape):
                params["n_pca"] = None
            if params["n_pca"] != self.n_pca:
                self.n_pca = params["n_pca"]
                reset_kernel = True
                del params["n_pca"]
        if "knn_dist" in params and params["knn_dist"] != self.knn_dist:
            self.knn_dist = params["knn_dist"]
            reset_kernel = True
            del params["knn_dist"]
        if "n_landmark" in params and params["n_landmark"] != self.n_landmark:
            if self.n_landmark is None or params["n_landmark"] is None:
                # need a different type of graph, reset entirely
                self._reset_graph()
            else:
                self._set_graph_params(n_landmark=params["n_landmark"])
            self.n_landmark = params["n_landmark"]
            del params["n_landmark"]

        # parameters that don't change the embedding
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
            self._set_graph_params(n_jobs=params["n_jobs"])
            del params["n_jobs"]
        if "random_state" in params:
            self.random_state = params["random_state"]
            self._set_graph_params(random_state=params["random_state"])
            del params["random_state"]
        if "verbose" in params:
            self.verbose = params["verbose"]
            logger.set_level(self.verbose)
            self._set_graph_params(verbose=params["verbose"])
            del params["verbose"]
            
        if reset_kernel or reset_embedding:
            self._reset_graph()

        self._set_graph_params(**params)


    def _set_graph_params(self, **params):
        try:
            self.G.set_params(**params)
        except AttributeError:
            # graph not defined
            pass
    
    def _reset_graph(self):
        self.G = None


# PyTorch Lightning Module wrapper

class DiffusionMapModule(LatentModule):
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = 42,
        knn: Optional[int] = 5,
        t: Union[int, str] = 15, # Can be an integer or 'auto'
        decay: Optional[int] = 40,
        n_pca: Optional[int] = None,
        n_landmark: Optional[int] = 2000,
        n_jobs: Optional[int] = -1,
        verbose = False,
        fit_fraction: float = 1.0,  # Fraction of data used for fitting
        **kwargs
    ):
        super().__init__(n_components=n_components, init_seed=random_state, **kwargs)
        self.fit_fraction = fit_fraction
        self.model = DiffusionMap(n_components=n_components, 
                                  random_state=random_state,
                                  knn=knn,
                                  t=t,
                                  decay=decay,
                                  n_pca=n_pca,
                                  n_landmark=n_landmark,
                                  n_jobs=n_jobs,
                                  verbose=verbose)

    def fit(self, x: Tensor) -> None:
        """Fits DiffusionMap on a subset of data."""
        x_np = x.detach().cpu().numpy()
        n_samples = x_np.shape[0]
        n_fit = max(1, int(self.fit_fraction * n_samples))  # Use only a fraction of the data
        self.model.fit(x_np[:n_fit])
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Transforms data using the fitted DiffusionMap model."""
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")
        
        x_np = x.detach().cpu().numpy()
        embedding = self.model.transform(x_np)
        return torch.tensor(embedding, device=x.device, dtype=x.dtype)

    def affinity_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns diffusion operator.

        The diffusion operator is computed during fit and represents the
        transition matrix of the diffusion process.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N diffusion operator matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")

        # Recompute diffusion operator from kernel
        K = np.asarray(self.model.G.kernel.todense())
        _, _, diff_op, _ = compute_dm(K, alpha=1.0)

        if ignore_diagonal:
            diff_op = diff_op - np.diag(np.diag(diff_op))
        return diff_op

    def kernel_matrix(self, ignore_diagonal: bool = False) -> np.ndarray:
        """
        Returns kernel matrix used to build diffusion operator.

        Args:
            ignore_diagonal: If True, set diagonal entries to zero. Default False.

        Returns:
            N×N kernel matrix.
        """
        if not self._is_fitted:
            raise RuntimeError("DiffusionMap model is not fitted yet. Call `fit` first.")

        K = np.asarray(self.model.G.kernel.todense())
        if ignore_diagonal:
            K = K - np.diag(np.diag(K))
        return K