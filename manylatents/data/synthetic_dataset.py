import os
import logging
import torch
from torch.utils.data import Dataset
import graphtools
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from scipy.stats import special_ortho_group
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from scipy.sparse.csgraph import shortest_path
from typing import Union, List, Optional, Dict, Tuple
from .precomputed_mixin import PrecomputedMixin
from ..utils.dla_tree_visualization import DLATreeGraphVisualizer


class SyntheticDataset(Dataset):
    """
    Base class for synthetic datasets.
    """

    def __init__(self):
        super().__init__()
        self.data = None
        self.metadata = None
        self.graph = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.metadata[idx] if self.metadata is not None else -1
        return {"data": torch.tensor(x, dtype=torch.float32), "metadata": torch.tensor(y, dtype=torch.long)}

    def get_labels(self):
        return self.metadata

    def get_data(self):
        return self.data

    def standardize_data(self):
        """
        Standardize data putting it in a unit box around the origin. (min_max normalization)
        This is necessary for quadtree type algorithms
        """
        X = self.data
        minx = np.min(self.data, axis=0)
        maxx = np.max(self.data, axis=0)
        self.std_X = (X - minx) / (maxx - minx)
        return self.std_X

    def rotate_to_dim(self, dim):
        """
        Rotate dataset to a different higher dimensionality.
        """
        self.rot_mat = special_ortho_group.rvs(dim)[: self.data.shape[1]]
        self.high_X = np.dot(self.data, self.rot_mat)
        return self.high_X

    def get_gt_dists(self):
        pass


class SwissRoll(SyntheticDataset, PrecomputedMixin):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.1,
        manifold_noise=0.1,
        width=10.0,
        random_state=42,
        rotate_to_dim=3,
        precomputed_path=None,
        mmap_mode=None,
    ):
        """
        Initialize a synthetic Swiss Roll dataset with parameters to control 
        the structure and noise characteristics. The distributions are generated
        by sampling points along a Swiss roll manifold, with Gaussian noise added to
        the points.

        Parameters:
        ----------
        n_distributions : int, default=100
            Number of independent Gaussian distributions to generate along the manifold.

        n_points_per_distribution : int, default=50
            Number of samples drawn from each Gaussian distribution.

        noise : float, default=0.1
            Standard deviation of isotropic Gaussian noise added to each data point 
            (i.e., global noise affecting all points).

        manifold_noise : float, default=0.1
            Controls the standard deviation within each local distribution on the 
            manifold (i.e., spread of each blob along the Swiss roll).

        width : float, default=10.0
            Width factor of the Swiss roll, affecting how "thick" the roll appears.

        random_state : int, default=42
            Seed for random number generator to ensure reproducibility of the synthetic data.

        rotate_to_dim : int, default=3
            The higher dimensionality of the space to which the manifold is rotated.
            Rotation is only applied when this value is greater than 3.
            For visualization purposes, the default of 3 means no rotation is applied.

        precomputed_path : str, optional
            Path to precomputed embeddings. If provided, the embeddings will be loaded from this path.
            If None, a new dataset will be generated.
        
        mmap_mode : str, optional
            Memory mapping mode for loading the dataset. If None, the dataset will be loaded into memory.
        """
        super().__init__()
        np.random.seed(random_state)
        rng = np.random.default_rng(random_state)

        self.mean_t = 3 * np.pi / 2 * (1 + 2 * rng.random((1, n_distributions)))
        # ground truth coordinate euclidean in (y,t) is geo on 3d

        # mean_y has shape (1, n_distributions) when width=1
        self.mean_y = width * rng.uniform(size=(1, n_distributions))

        # t_noise.shape: (n_distributions, n_points_per_distribution)
        t_noise = manifold_noise * rng.normal(size=(n_distributions, n_points_per_distribution))

        # y_noise.shape: (n_distributions, n_points_per_distribution)
        y_noise = width * manifold_noise * rng.normal(size=(n_distributions, n_points_per_distribution))
        ts = np.reshape(t_noise + self.mean_t.T, -1)  # shape (5000,)
        ys = np.reshape(y_noise + self.mean_y.T, -1)  # shape (5000,)
        self.ys = ys

        xs = ts * np.cos(ts)
        zs = ts * np.sin(ts)
        X = np.stack((xs, ys, zs))  # shape (3, 5000)
        noise_term = noise * rng.normal(size=(3, n_distributions * n_points_per_distribution))
        X = X + noise_term
        # load precomputed embeddings or generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:   
            self.data = X.T  # shape (5000, 3)
            if rotate_to_dim > 3:
                self.data = self.rotate_to_dim(rotate_to_dim)

        self.ts = np.squeeze(ts)  # (5000,)
        self.metadata = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        
        self.metadata = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        self.metadata = np.argmax(self.metadata,-1)

    def _unroll_t(self, t):
        t = t.flatten()  # (100,)
        return 0.5 * ((np.sqrt(t**2 + 1) * t) + np.arcsinh(t)).reshape(
            1, -1
        )  # (1, 100)

    def get_gt_dists(self):
        u_t = self._unroll_t(self.ts)  # (1, 5000)
        true_coords = np.concatenate(
            (u_t, self.ys[None, ...])
        ).T  # (5000, 2) This is a 2D space
        geodesic_dist = pairwise_distances(true_coords, metric="euclidean") # (5000,5000)
        return geodesic_dist

    def get_graph(self):
        """Create a graphtools graph if does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.data, use_pygsp=True)
        return self.graph
    

class SaddleSurface(SyntheticDataset):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.1,
        manifold_noise=0.1,
        a=1.0,
        b=1.0,
        random_state=42,
        rotate_to_dim=3,
    ):
        """
        Initialize a synthetic Saddle Surface dataset with tunable parameters for geometry and noise.

        Parameters
        ----------
        n_distributions : int, default=100
            Number of independent Gaussian distributions sampled across the saddle surface manifold.

        n_points_per_distribution : int, default=50
            Number of data points drawn from each Gaussian distribution.

        noise : float, default=0.1
            Global isotropic Gaussian noise added to all data points.

        manifold_noise : float, default=0.1
            Local noise controlling spread within each distribution on the manifold surface.

        a : float, default=1.0
            Coefficient scaling the x-direction of the saddle surface (z = a * x^2 - b * y^2).

        b : float, default=1.0
            Coefficient scaling the y-direction curvature of the saddle surface (z = a * x^2 - b * y^2).

        random_state : int, default=42
            Seed for the random number generator to ensure reproducibility.

        rotate_to_dim : int, default=3
            The higher dimensionality of the space to which the manifold is rotated.
            Rotation is only applied when this value is greater than 3.
            For visualization purposes, the default of 3 means no rotation is applied.
        """
        super().__init__()
        np.random.seed(random_state)
        self.n_distributions = n_distributions
        self.n_points_per_distribution = n_points_per_distribution
        self.noise = noise
        self.manifold_noise = manifold_noise
        self.a = a
        self.b = b
        self.random_state = random_state

        self.u_centers = np.random.uniform(-2, 2, n_distributions)
        self.v_centers = np.random.uniform(-2, 2, n_distributions)

        self.u_samples = []
        self.v_samples = []
        self.point_sets = []

        gt_points = self._generate_gaussian_blobs()
        self.data = gt_points
        if rotate_to_dim > 3:
            self.data = self.rotate_to_dim(rotate_to_dim)
        self.data = self._apply_noise(self.data)
        self.gt_points = gt_points

        self.metadata = np.repeat(
            np.eye(self.n_distributions), self.n_points_per_distribution, axis=0
        )
        self.metadata = np.argmax(self.metadata,-1)

    def _generate_gaussian_blobs(self):
        """Generate Gaussian blobs in the parameter space of the saddle surface."""
        for i in range(self.n_distributions):
            u_blob = np.random.normal(
                self.u_centers[i], self.manifold_noise, self.n_points_per_distribution
            )
            v_blob = np.random.normal(
                self.v_centers[i], self.manifold_noise, self.n_points_per_distribution
            )
            self.u_samples.append(u_blob)
            self.v_samples.append(v_blob)
            self.point_sets.append(self._saddle_to_cartesian(u_blob, v_blob))
        X = np.concatenate(self.point_sets)
        return X

    def _apply_noise(self, X):
        """Add noise to the points in the Cartesian space to simulate noisy data."""
        X = X + np.random.normal(0, self.noise, X.shape)
        return X

    def _saddle_to_cartesian(self, u, v):
        """Convert saddle coordinates to Cartesian coordinates."""
        x = u
        y = v
        z = self.a * u**2 - self.b * v**2
        return np.stack((x, y, z), axis=-1)

    def get_gt_dists(self):
        """Compute pairwise geodesic distances for a specific distribution using a surface-based distance."""
        points = self.gt_points
        num_points = points.shape[0]
        distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                distances[i, j] = self._surface_geodesic_distance(points[i], points[j])
                distances[j, i] = distances[i, j]

        return distances

    def _surface_geodesic_distance(self, p1, p2):
        """Approximate the geodesic distance between two points on the saddle surface."""
        u1, v1 = p1[0], p1[1]
        u2, v2 = p2[0], p2[1]
        distance = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)

        return distance
    
    def get_graph(self):
        """Create a graphtools graph if does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.data, use_pygsp=True)
        return self.graph


import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GaussianBlobs(SyntheticDataset, PrecomputedMixin):
    """
    Gaussian K blobs synthetic dataset that integrates with the synthetic dataset framework.
    """

    def __init__(self,
                 n_samples: Union[int, List[int]] = 100,
                 n_features: int = 2,
                 centers: Optional[Union[int, np.ndarray, List[List[float]]]] = None,
                 cluster_std: Union[float, List[float]] = 1.0,
                 center_box: Tuple[float, float] = (-10.0, 10.0),
                 shuffle: bool = True,
                 random_state: Optional[int] = 42,
                 return_centers: bool = False,
                 precomputed_path: Optional[str] = None,
                 mmap_mode: Optional[str] = None):
        """
        Initialize a synthetic Gaussian Blobs dataset using sklearn's make_blobs.

        Parameters:
        ----------
        n_samples : int or list of int, default=100
            Number of samples to generate. If int, total samples. If list, samples per center.

        n_features : int, default=2
            Number of features (dimensionality) for each sample.

        centers : int, array-like or None, default=None
            Number of centers to generate, or fixed center locations.

        cluster_std : float or list of float, default=1.0
            Standard deviation of clusters.

        center_box : tuple of float, default=(-10.0, 10.0)
            Bounding box for randomly generated cluster centers.

        shuffle : bool, default=True
            Whether to shuffle the samples.

        random_state : int, default=42
            Random seed for reproducibility.

        return_centers : bool, default=False
            Whether to return the cluster centers.

        precomputed_path : str, optional
            Path to precomputed embeddings.
        
        mmap_mode : str, optional
            Memory mapping mode for loading precomputed data.
        """
        super().__init__()
        
        logger.info("Generating Gaussian blobs")

        # Generate the blob data using sklearn
        result = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=tuple(center_box),
            shuffle=shuffle,
            random_state=random_state,
            return_centers=return_centers
        )

        if return_centers:
            self.X, self.y, self.centers = result
        else:
            self.X, self.y = result
            self.centers = None

        # Load precomputed embeddings or use generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:
            self.data = self.X

        self.metadata = self.y

        # Create DataFrame for compatibility with your existing interface
        self.FinalData = pd.DataFrame({
            'sample_id': [f"sample_{i}" for i in range(len(self.X))],
            'label': self.y
        }).set_index("sample_id")

    def get_data(self) -> np.ndarray:
        """Return the feature data."""
        return self.data

    def get_labels(self) -> np.ndarray:
        """Return the cluster labels."""
        return self.metadata

    def get_FinalData(self) -> pd.DataFrame:
        """Return the DataFrame with sample IDs and labels."""
        return self.FinalData

    def get_centers(self) -> Optional[np.ndarray]:
        """Return cluster centers if available."""
        return self.centers

    def get_gt_dists(self):
        """
        Compute ground truth distances between points.
        
        For Gaussian blobs, we use Euclidean distance as the ground truth
        since the data exists in Euclidean space.
        """
        return pairwise_distances(self.data, metric="euclidean")

    def get_graph(self):
        """Create a graphtools graph if it does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.data, use_pygsp=True)
        return self.graph

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a single sample and its metadata.
        
        Returns dict with 'data' and 'metadata' keys for consistency
        with the base SyntheticDataset class.
        """
        x = self.data[idx]
        y = self.metadata[idx] if self.metadata is not None else -1
        return {"data": torch.tensor(x, dtype=torch.float32), "metadata": torch.tensor(y, dtype=torch.long)}
    
class DLAtree(SyntheticDataset, PrecomputedMixin):
    def __init__(
        self,
        n_dim: int = 3,
        n_branch: int = 20,
        branch_lengths: Union[List[int], int, None] = None,  # List of branch lengths, one per branch
        rand_multiplier: float = 2,
        gap_multiplier: float = 0,
        random_state: int = 42,
        sigma: float = 4,
        disconnect_branches: Optional[List[int]] = [5,15],  # Branch indices to disconnect
        sampling_density_factors: Optional[Dict[int, float]] = None,  # Reduce density of certain branches
        precomputed_path: Optional[str] = None,
        mmap_mode: Optional[str] = None,
    ):

        """
        Generate a Diffusion-Limited Aggregation (DLA) tree with optional branch disconnections.

        Parameters
        ----------
        n_dim : int, default=3
            Number of dimensions for each point in the tree.

        n_branch : int, default=20
            Number of branches in the tree.

        branch_lengths : int or list of int or None, default=100
            Length of each branch. If an int is provided, all branches will have the same length.

        rand_multiplier : float, default=2.0
            Scaling factor for random movement along the tree.

        gap_multiplier : float, default=0.0
            Scaling factor for the gap added when disconnecting branches.

        random_state : int, default=37
            Seed for the random number generator to ensure reproducibility.

        sigma : float, default=4.0
            Standard deviation of Gaussian noise added to all data points.

        disconnect_branches : list of int or None, optional
            Indices of branches to disconnect from the main structure.

        sampling_density_factors : dict of int to float or None, optional
            Dictionary mapping branch index to sampling reduction factor (e.g., 0.5 keeps 50% of points).
        
        precomputed_path : str, optional
            Path to precomputed embeddings. If provided, the embeddings will be loaded from this path.
            If None, a new dataset will be generated.
        
        mmap_mode : str, optional
            Memory mapping mode for loading the dataset. If None, the dataset will be loaded into memory.
        """
        super().__init__()
        # Generate the DLA data
        data, data_gt, metadata = self._make_sim_data(
            n_dim=n_dim,
            n_branch=n_branch,
            branch_lengths=branch_lengths,
            rand_multiplier=rand_multiplier,
            gap_multiplier=gap_multiplier,
            random_state=random_state,
            sigma=sigma,
            disconnect_branches=disconnect_branches or [],
            sampling_density_factors=sampling_density_factors,
        )
        # Load precomputed embeddings or generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:
            self.data = data  # noisy data
        self.data_gt = data_gt  # ground truth tree
        self.metadata = metadata
        self.n_dim = n_dim
        
    def get_gt_dists(self):
        """
        Compute geodesic distances as shortest paths over the tree graph
        """
        if self.graph is None:
            self.get_graph()
        geodesic_dist = shortest_path(csgraph=self.graph.tocsr(), directed=False)
        return geodesic_dist

    def get_graph(self):
        """
        Build a sparse adjacency graph connecting neighboring points along the tree structure
        """
        if self.graph is None:
            n_points = self.data_gt.shape[0]
            adj_matrix = sp.lil_matrix((n_points, n_points))

            for i in range(n_points - 1):
                distance = np.linalg.norm(self.data_gt[i] - self.data_gt[i + 1])
                adj_matrix[i, i + 1] = distance
                adj_matrix[i + 1, i] = distance

            self.graph = adj_matrix
        return self.graph
    
    ### same method of get_geodesic and get_graph as DeMAP but might not work well on DLA tree
    # def get_geodesic(self):
    #     # Compute geodesic distances as shortest paths over the graphtools graph
    #     if self.graph is None:
    #         self.get_graph()
    #     adj_matrix = self.graph.graph  # extract adjacency matrix from Graph object
    #     geodesic_dist = shortest_path(csgraph=adj_matrix.tocsr(), directed=False)
    #     return geodesic_dist

    # def get_graph(self):
    #     # Build and return a graphtools.Graph object (DeMAP style) on noisy data
    #     if self.graph is None:
    #         import graphtools
    #         self.graph = graphtools.Graph(self.data_gt, knn=10, decay=None)
    #     return self.graph

    def _make_sim_data(
        self,
        n_dim: int,
        n_branch: int,
        branch_lengths: int,
        rand_multiplier: float,
        gap_multiplier: float,
        random_state: int,
        sigma: float,
        disconnect_branches: list,
        sampling_density_factors: dict,
    ):
        return self._gen_dla_with_disconnectivity(
            n_dim=n_dim,
            n_branch=n_branch,
            branch_lengths=branch_lengths,
            rand_multiplier=rand_multiplier,
            gap_multiplier=gap_multiplier,
            random_state=random_state,
            sigma=sigma,
            disconnect_branches=disconnect_branches,
            sampling_density_factors=sampling_density_factors,
        )

    def _gen_dla_with_disconnectivity(
        self,
        n_dim: int,
        n_branch: int,
        branch_lengths,
        rand_multiplier: float,
        gap_multiplier: float,
        random_state: int,
        sigma: float,
        disconnect_branches: list,
        sampling_density_factors: dict,
    ):
        """
        Generates a Diffusion-Limited Aggregation (DLA) tree with optional branch disconnections.
        """
        np.random.seed(random_state)

        # Default branch lengths if none are provided
        if branch_lengths is None:
            branch_lengths = [100] * n_branch
        elif isinstance(branch_lengths, int):
            branch_lengths = [branch_lengths] * n_branch
        elif len(branch_lengths) != n_branch:
            raise ValueError("The length of 'branch_lengths' must match 'n_branch'.")

        # Initialize the first branch
        M_gt = np.cumsum(-1 + rand_multiplier * np.random.rand(branch_lengths[0], n_dim), axis=0)
        M = M_gt.copy()  # Start with ground truth dataset
        branch_start_indices = [0]  # Keep track of where each branch starts

        for i in range(1, n_branch):
            ind = np.random.randint(branch_start_indices[i - 1], branch_start_indices[i - 1] + branch_lengths[i - 1])

            # Create the ground truth branch first
            new_branch_gt = np.cumsum(
                -1 + rand_multiplier * np.random.rand(branch_lengths[i], n_dim), axis=0
            )
            new_branch_gt += M_gt[ind]

            # Create the potentially disconnected branch
            new_branch = new_branch_gt.copy()
            if i in disconnect_branches:
                jump = np.random.normal(gap_multiplier, 0.1, n_dim)  # Jump offset
                new_branch += jump  # Apply the jump to all points in the branch

                # Check if the jump places the branch too close to another branch
                distances = np.linalg.norm(M - new_branch[0], axis=1)
                if np.min(distances) < rand_multiplier:
                    raise ValueError(f"Jump for branch {i} is too close to another branch. Adjust gap_multiplier.")

            M_gt = np.concatenate([M_gt, new_branch_gt])
            M = np.concatenate([M, new_branch])
            branch_start_indices.append(M.shape[0] - branch_lengths[i])

        # Reduce sampling density for certain branches
        if sampling_density_factors:
            mask = np.ones(M.shape[0], dtype=bool)
            for branch_idx, factor in sampling_density_factors.items():
                start_idx = branch_start_indices[branch_idx]
                end_idx = start_idx + branch_lengths[branch_idx]
                branch_points = np.arange(start_idx, end_idx)
                keep_points = np.random.choice(branch_points, int(len(branch_points) * factor), replace=False)
                mask[branch_points] = False  # Remove points
                mask[keep_points] = True  # Retain selected points
            M = M[mask]
            M_gt = M_gt[mask]  # Apply the same mask to the ground truth
        else:
            mask = None

        # Add noise
        noise = np.random.normal(0, sigma, M.shape)
        M = M + noise
        M_gt = M_gt + noise

        C = np.array(
            [i for branch_idx, branch_len in enumerate(branch_lengths)
             for i in [branch_idx] * branch_len]
        )
        if mask is not None:
            C = C[mask]

        return M, M_gt, C


class Torus(SyntheticDataset, PrecomputedMixin):
    def __init__(
        self,
        n_points=5000,
        noise=0.1,
        major_radius=3.0,
        minor_radius=1.0,
        random_state=42,
        rotate_to_dim=3,
        precomputed_path=None,
        mmap_mode=None,
    ):
        """
        Initialize a synthetic Torus dataset with uniformly distributed points.

        Parameters:
        ----------
        n_points : int, default=5000
            Total number of points to generate on the torus surface.

        noise : float, default=0.1
            Standard deviation of isotropic Gaussian noise added to each data point.

        major_radius : float, default=3.0
            Major radius of the torus (distance from center to tube center).

        minor_radius : float, default=1.0
            Minor radius of the torus (radius of the tube).

        random_state : int, default=42
            Seed for random number generator to ensure reproducibility.

        rotate_to_dim : int, default=3
            The higher dimensionality of the space to which the manifold is rotated.
            Rotation is only applied when this value is greater than 3.

        precomputed_path : str, optional
            Path to precomputed embeddings. If provided, the embeddings will be loaded from this path.
        
        mmap_mode : str, optional
            Memory mapping mode for loading the dataset. If None, the dataset will be loaded into memory.
        """
        super().__init__()
        np.random.seed(random_state)
        rng = np.random.default_rng(random_state)

        # Generate uniformly distributed angles
        self.theta_all = 2 * np.pi * rng.random(n_points)  # [0, 2π]
        self.phi_all = 2 * np.pi * rng.random(n_points)    # [0, 2π]
        
        # Convert to Cartesian coordinates
        x = (major_radius + minor_radius * np.cos(self.phi_all)) * np.cos(self.theta_all)
        y = (major_radius + minor_radius * np.cos(self.phi_all)) * np.sin(self.theta_all)
        z = minor_radius * np.sin(self.phi_all)
        
        X = np.stack((x, y, z), axis=-1)  # shape (n_points, 3)
        
        # Add global noise
        noise_term = noise * rng.normal(size=X.shape)
        X = X + noise_term
        
        # Store parameters
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        
        # Load precomputed embeddings or use generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:
            self.data = X
            if rotate_to_dim > 3:
                self.data = self.rotate_to_dim(rotate_to_dim)

        # Create simple metadata
        self.metadata = np.zeros(n_points, dtype=int)

    def get_gt_dists(self):
        """
        Compute geodesic distances on the torus surface using parameter space distances.
        """
        # Vectorized computation using broadcasting
        theta = self.theta_all[:, np.newaxis]  # shape (n_points, 1)
        phi = self.phi_all[:, np.newaxis]      # shape (n_points, 1)
        
        # Compute pairwise differences with broadcasting
        theta_diff = np.abs(theta - self.theta_all)  # shape (n_points, n_points)
        phi_diff = np.abs(phi - self.phi_all)        # shape (n_points, n_points)
        
        # Apply periodic boundary conditions
        theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)
        phi_diff = np.minimum(phi_diff, 2 * np.pi - phi_diff)
        
        # Compute geodesic distances
        distances = np.sqrt((self.major_radius * theta_diff)**2 + (self.minor_radius * phi_diff)**2)
        
        return distances

    def get_graph(self):
        """Create a graphtools graph if does not exist."""
        if self.graph is None:
            self.graph = graphtools.Graph(self.data, use_pygsp=True)
        return self.graph

class DLATreeFromGraph(SyntheticDataset, PrecomputedMixin):
    def __init__(
        self,
        graph_edges,  # List of (from_node, to_node, edge_id, length)
        n_dim: int = 50,
        rand_multiplier: float = 2.0,
        random_state: int = 42,
        sigma: float = 0.5,
        save_graph_viz: bool = True,
        save_dir: str = "outputs",
        precomputed_path: Optional[str] = None,
        mmap_mode: Optional[str] = None,
        # Gap functionality: simply exclude these edge_ids from data generation
        excluded_edges: Optional[List] = None,
    ):
        """
        Generate a DLA (Diffusion-Limited Aggregation) tree from explicit graph topology.
        
        This class creates synthetic tree-structured data by following a user-defined graph
        topology. Each edge in the graph becomes a population/branch in the generated dataset,
        with the number of samples determined by the edge's length parameter.
        
        The algorithm generates data by:
        1. Traversing graph edges in the order specified
        2. For each edge, generating samples using a DLA random walk process
        3. Creating distinct populations labeled by edge_id
        4. Optionally excluding certain edges to create "gaps" in the data
        5. Renumbering remaining edges to be sequential (1, 2, 3, ...)
        
        Parameters:
        -----------
        graph_edges : list of tuples
            List of (from_node, to_node, edge_id, length) where:
            - from_node, to_node: Node identifiers (strings or integers)
            - edge_id: Integer identifier for the edge (becomes population label)
            - length: Number of samples to generate for this population/edge
            
        n_dim : int, default=50
            Dimensionality of the generated data space
            
        rand_multiplier : float, default=2.0
            Controls randomness in the DLA random walk process. Higher values create
            more branching/spreading behavior.
            
        random_state : int, default=42
            Seed for reproducible random number generation
            
        sigma : float, default=0.5
            Standard deviation of Gaussian noise added to all generated points
            
        save_graph_viz : bool, default=True
            Whether to generate and save graph topology visualizations
            
        save_dir : str, default="outputs"
            Directory where visualizations will be saved
            
        excluded_edges : list, optional
            List of edge_ids to exclude from data generation (creates "gaps").
            These edges are skipped during data generation but shown as dashed lines
            in the debug visualization. Useful for simulating missing populations
            or discontinuities in the tree.
            
        precomputed_path : str, optional
            Path to precomputed embeddings. If provided, loads data instead of generating.
            
        mmap_mode : str, optional
            Memory mapping mode for loading precomputed data.
        
        Algorithm Details:
        ------------------
        1. **Graph Parsing**: Build adjacency structure from graph_edges
        2. **Root Detection**: Find root node (appears in from_node but not to_node)
        3. **Sequential Generation**: For each edge in order:
           - Generate DLA random walk with specified length
           - Assign edge_id as population label
           - Connect to previous branch endpoints
        4. **Gap Application**: Remove samples from excluded edges
        5. **Label Renumbering**: Create sequential labels (1,2,3...) for remaining populations
        6. **Noise Addition**: Add Gaussian noise with specified sigma
        
        Output Data Structure:
        ----------------------
        - data: High-dimensional point cloud (n_samples, n_dim)
        - metadata: Population labels for each sample (renumbered to be sequential)
        - edge_renumbering: Mapping from original edge_ids to new sequential ids
        - original_excluded_edges: Set of edge_ids that were excluded
        
        Visualization:
        --------------
        Automatically generates graph topology visualization showing:
        - Solid colored edges: Data-containing populations
        - Dashed gray edges: Excluded/gap edges (no data)
        - Edge labels: Sequential population numbers (1, 2, 3, ...)
        - Node structure: Tree connectivity based on graph_edges
        
        Example Usage:
        --------------
        ```python
        # Define tree topology
        graph_edges = [
            (1, 2, 1, 300),    # Main trunk: 300 samples, edge 1
            (2, 3, 2, 300),    # Continuation: 300 samples, edge 2
            (2, 4, 3, 150),    # Side branch: 150 samples, edge 3
            (2, 5, 4, 75),     # Cross-link: 75 samples, edge 4
        ]
        
        # Create gaps by excluding certain edges
        excluded_edges = [2, 4]  # Remove edges 2 and 4
        
        # Generate dataset
        dataset = DLATreeFromGraph(
            graph_edges=graph_edges,
            excluded_edges=excluded_edges,
            n_dim=100,
            sigma=0.5
        )
        
        # Access data and labels
        data = dataset.data                    # (samples, 100)
        labels = dataset.metadata              # Sequential labels: [1, 1, ..., 2, 2, ...]
        renumbering = dataset.edge_renumbering # {1: 1, 3: 2} (edges 2,4 excluded)
        ```
        
        Notes:
        ------
        - Excluded edges create natural discontinuities in the generated tree
        - Final population labels are always sequential (1, 2, 3, ...) after renumbering
        - Graph topology visualization includes both data edges and gap edges
        - Root node is automatically detected as the node with no incoming edges
        """
        super().__init__()
        self.graph_edges = graph_edges
        self.excluded_edges = set(excluded_edges or [])  # Convert to set for fast lookup
        self.n_dim = n_dim
        self.rand_multiplier = rand_multiplier
        self.random_state = random_state
        self.sigma = sigma
        self.save_graph_viz = save_graph_viz
        self.save_dir = save_dir
        
        # Generate the data using simplified approach
        data, graph, metadata = self._generate_simplified()
        
        # Load precomputed or use generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:
            self.data = data

        self.graph = graph

        self.metadata = metadata
        
        # Generate and save visualization
        if save_graph_viz:
            visualizer = DLATreeGraphVisualizer(
                graph_edges=self.graph_edges,
                excluded_edges=self.excluded_edges,
                edge_renumbering=getattr(self, 'edge_renumbering', None),
                original_excluded_edges=getattr(self, 'original_excluded_edges', set()),
                random_state=self.random_state,
                save_dir=self.save_dir
            )
            visualizer.visualize_and_save_graph()
            # Also generate sample graph visualization
            if hasattr(self, 'sample_graph'):
                visualizer.visualize_sample_graph(self.sample_graph, save_path="outputs/sample_graph.png")

    def _build_complete_structure(self):
        """
        Build the complete DLA tree structure in 3 clear steps:
        1. Convert config graph to complete sample adjacency matrix
        1.5. Generate visualizations (PNG plots)
        2. Generate complete data using DLA branches
        3. Subset adjacency and data to visible nodes only
        """
        np.random.seed(self.random_state)

        # Step 1: Build complete adjacency matrix from config graph topology
        self._build_complete_adjacency_matrix()

        # Step 1.5: Generate visualizations now that we have the topology
        if hasattr(self, 'save_graph_viz') and self.save_graph_viz:
            from manylatents.utils.dla_tree_visualization import DLATreeGraphVisualizer
            visualizer = DLATreeGraphVisualizer(
                graph_edges=self.graph_edges,
                excluded_edges=self.excluded_edges,
                original_excluded_edges=self.excluded_edges,
                save_dir=getattr(self, 'save_dir', 'outputs')
            )
            # Create topology-based visualization
            visualizer.visualize_and_save_graph()

        # Step 2: Generate complete data using DLA branches
        self._generate_complete_data()

        # Step 3: Subset to visible nodes only
        self._subset_to_visible_nodes()

    def _build_complete_adjacency_matrix(self):
        """
        Step 1: Convert the config graph (edges) to a complete sample-level adjacency matrix.
        All edge weights are 1 since we only care about topological distance.
        """
        # Calculate total number of samples across all edges
        total_samples = sum(length for _, _, _, length in self.graph_edges)

        # Create adjacency matrix (all weights = 1 for topological distance)
        import scipy.sparse as sp
        adj_matrix = sp.lil_matrix((total_samples, total_samples))

        # Track sample ranges for each edge
        sample_idx = 0
        edge_sample_ranges = {}  # edge_id -> (start_idx, end_idx)
        sample_to_edge = {}      # sample_idx -> edge_id

        for from_node, to_node, edge_id, length in self.graph_edges:
            start_idx = sample_idx
            end_idx = sample_idx + length
            edge_sample_ranges[edge_id] = (start_idx, end_idx)

            # Connect consecutive samples within this edge
            for i in range(start_idx, end_idx - 1):
                adj_matrix[i, i + 1] = 1.0
                adj_matrix[i + 1, i] = 1.0  # symmetric
                sample_to_edge[i] = edge_id
            sample_to_edge[end_idx - 1] = edge_id  # last sample

            sample_idx = end_idx

        # Connect edges at shared nodes (where edges meet)
        node_edge_boundaries = {}  # node_id -> list of (edge_id, boundary_sample_idx)

        for from_node, to_node, edge_id, length in self.graph_edges:
            start_idx, end_idx = edge_sample_ranges[edge_id]
            first_sample = start_idx
            last_sample = end_idx - 1

            # Track which samples are at node boundaries
            if from_node not in node_edge_boundaries:
                node_edge_boundaries[from_node] = []
            if to_node not in node_edge_boundaries:
                node_edge_boundaries[to_node] = []

            node_edge_boundaries[from_node].append((edge_id, first_sample))
            node_edge_boundaries[to_node].append((edge_id, last_sample))

        # Connect samples at shared nodes (all edges meeting at a node are connected)
        for node_id, edge_boundaries in node_edge_boundaries.items():
            if len(edge_boundaries) > 1:
                # Connect all boundary samples at this node
                boundary_samples = [sample_idx for _, sample_idx in edge_boundaries]
                for i in range(len(boundary_samples)):
                    for j in range(i + 1, len(boundary_samples)):
                        sample_i, sample_j = boundary_samples[i], boundary_samples[j]
                        adj_matrix[sample_i, sample_j] = 1.0
                        adj_matrix[sample_j, sample_i] = 1.0

        # Store complete adjacency matrix and metadata
        self.adj_matrix_complete = adj_matrix.tocsr()
        self.edge_sample_ranges = edge_sample_ranges
        self.sample_to_edge = sample_to_edge

        # Create complete metadata (which edge each sample belongs to)
        metadata_complete = []
        for from_node, to_node, edge_id, length in self.graph_edges:
            metadata_complete.extend([edge_id] * length)
        self.metadata_complete = np.array(metadata_complete)

    def _generate_complete_data(self):
        """
        Step 2: Generate the actual DLA branch data in the SAME ORDER as adjacency matrix.
        The adjacency matrix was built with samples ordered by self.graph_edges iteration.
        """
        # Find root node (appears as from_node but never as to_node)
        all_nodes = set()
        for from_node, to_node, _, _ in self.graph_edges:
            all_nodes.add(from_node)
            all_nodes.add(to_node)

        to_nodes = {to_node for from_node, to_node, _, _ in self.graph_edges}
        root_node = next((node for node in all_nodes if node not in to_nodes), sorted(all_nodes)[0])

        # Create orthogonal subspaces for branching nodes
        node_outgoing_edges = {}
        for from_node, to_node, edge_id, length in self.graph_edges:
            if from_node not in node_outgoing_edges:
                node_outgoing_edges[from_node] = []
            node_outgoing_edges[from_node].append((to_node, edge_id))

        node_subspaces = {}
        for node, outgoing_edges in node_outgoing_edges.items():
            if len(outgoing_edges) > 1:
                node_subspaces[node] = self._create_orthogonal_subspaces(
                    n_directions=len(outgoing_edges), n_dim=self.n_dim, node_id=node)

        # Generate branches in dependency order to get node positions
        node_positions = {root_node: np.zeros(self.n_dim)}
        edge_to_branch_data = {}  # edge_id -> branch_data

        processed_edges = set()
        iteration = 0
        max_iterations = len(self.graph_edges) * 2

        while len(processed_edges) < len(self.graph_edges) and iteration < max_iterations:
            iteration += 1
            made_progress = False

            for from_node, to_node, edge_id, length in self.graph_edges:
                if (from_node, to_node, edge_id) in processed_edges or from_node not in node_positions:
                    continue

                # Get starting position and subspace constraints
                start_pos = node_positions[from_node]
                allowed_dims = None
                if from_node in node_subspaces:
                    outgoing_edges = node_outgoing_edges[from_node]
                    for i, (_, out_edge_id) in enumerate(outgoing_edges):
                        if out_edge_id == edge_id:
                            allowed_dims = node_subspaces[from_node][i]
                            break

                # Generate DLA branch
                branch_data = self._generate_dla_branch_topology_aware(
                    start_pos=start_pos, length=length, edge_id=edge_id, allowed_dims=allowed_dims)

                edge_to_branch_data[edge_id] = branch_data
                node_positions[to_node] = branch_data[-1].copy()
                processed_edges.add((from_node, to_node, edge_id))
                made_progress = True

        # Now combine data in the SAME ORDER as we built the adjacency matrix
        all_data = []
        for from_node, to_node, edge_id, length in self.graph_edges:
            branch_data = edge_to_branch_data[edge_id]
            all_data.append(branch_data)

        # Combine all data
        M_complete = np.vstack(all_data) if all_data else np.empty((0, self.n_dim))

        # Add noise if specified
        if self.sigma > 0:
            M_complete += np.random.normal(0, self.sigma, M_complete.shape)

        # Store complete data
        self.M_complete = M_complete

    def _subset_to_visible_nodes(self):
        """
        Step 3: Extract only the visible samples (excluding gap edges).
        """
        # Store original excluded edges for visualization
        self.original_excluded_edges = self.excluded_edges.copy()

        # Create mask for visible samples (exclude gap edges)
        self.visible_mask = ~np.isin(self.metadata_complete, list(self.excluded_edges))
        self.visible_indices = np.where(self.visible_mask)[0]

        # Extract visible data and metadata
        M_visible = self.M_complete[self.visible_indices]
        metadata_visible = self.metadata_complete[self.visible_indices]

        # Renumber visible edge labels to be sequential (1, 2, 3, ...)
        if self.excluded_edges:
            unique_visible_edges = sorted(set(metadata_visible))
            self.edge_renumbering = {old_id: new_id for new_id, old_id in enumerate(unique_visible_edges, 1)}
            metadata_visible = np.array([self.edge_renumbering[old_id] for old_id in metadata_visible])
        else:
            self.edge_renumbering = None

        # Store final dataset
        self.data = M_visible
        self.metadata = metadata_visible

    def _generate_simplified(self):
        """
        Simplified generation approach that builds complete structure first,
        then extracts visible subset.
        """
        # Generate complete structure with full topology and connectivity
        self._build_complete_structure()

        # Return the final dataset components
        # Note: self.data, self.metadata are set in _build_complete_structure
        import scipy.sparse as sp
        placeholder_graph = sp.lil_matrix((len(self.data), len(self.data)))
        return self.data, placeholder_graph, self.metadata

    def visualize_sample_graph(self, save_path="debug_outputs/sample_graph.png", max_nodes=500):
        """
        Visualize the sample-level graph where nodes are samples and edges connect them.

        This method is now a wrapper around the DLATreeGraphVisualizer class.
        """
        if not hasattr(self, 'sample_graph'):
            print("No sample graph available. Run generation first.")
            return

        visualizer = DLATreeGraphVisualizer(
            graph_edges=self.graph_edges,
            excluded_edges=self.excluded_edges,
            edge_renumbering=getattr(self, 'edge_renumbering', None),
            original_excluded_edges=getattr(self, 'original_excluded_edges', set()),
            random_state=self.random_state,
            save_dir=self.save_dir
        )
        visualizer.visualize_sample_graph(self.sample_graph, save_path, max_nodes)


    def _create_orthogonal_subspaces(self, n_directions, n_dim, node_id):
        """
        Create orthogonal feature subspaces for branches emanating from a single node.
        
        This ensures that multiple edges starting from the same node diffuse in 
        completely orthogonal feature subspaces, guaranteeing they remain distinct.
        
        Algorithm:
        1. Divide n_dim features into n_directions orthogonal subspaces
        2. Each branch gets assigned a unique subspace of size ~(n_dim/n_directions)
        3. Branches can only diffuse within their assigned subspace
        4. Fixed features outside subspace keep branches separated
        
        Parameters:
        -----------
        n_directions : int
            Number of orthogonal subspaces needed (degree of the node)
        n_dim : int  
            Total dimensionality of the space
        node_id : int
            Node identifier for debugging
            
        Returns:
        --------
        list of np.array : List of feature indices for each branch, each indicating
                          which dimensions that branch is allowed to vary in
        
        Example:
        --------
        For node N2 with 4 outgoing edges and n_dim=100:
        - Branch E1: can diffuse in features [0:25], others fixed
        - Branch E2: can diffuse in features [25:50], others fixed  
        - Branch E6: can diffuse in features [50:75], others fixed
        - Branch E12: can diffuse in features [75:100], others fixed
        This ensures perfect orthogonality by construction.
        """
        if n_directions == 1:
            # Single direction - can use all dimensions
            return [np.arange(n_dim)]
        
        # Divide dimensions roughly equally among branches
        subspace_size = n_dim // n_directions
        remaining_dims = n_dim % n_directions
        
        logging.debug(f"Creating {n_directions} orthogonal subspaces for node {node_id}: ~{subspace_size} dims each, {remaining_dims} distributed")
        
        subspaces = []
        current_start = 0
        
        for i in range(n_directions):
            # Give extra dimensions to first few branches if needed
            current_size = subspace_size + (1 if i < remaining_dims else 0)
            current_end = current_start + current_size
            
            # Assign this range of features to this branch
            subspace_indices = np.arange(current_start, current_end)
            subspaces.append(subspace_indices)
            
            logging.debug(f"Branch {i}: diffuses in features [{current_start}:{current_end}] ({current_size} dims)")
            current_start = current_end
        
        return subspaces

    def _generate_dla_branch_topology_aware(self, start_pos, length, edge_id, allowed_dims=None):
        """
        Generate a DLA branch that respects graph topology with orthogonal subspace constraints.
        
        This method creates a natural DLA random walk while ensuring proper branching
        for nodes with multiple outgoing edges. When multiple edges emanate from the
        same node, each branch is restricted to diffuse only in its assigned orthogonal
        subspace, ensuring perfect separation by construction.
        
        Algorithm:
        1. Start at the junction node position (start_pos)
        2. If allowed_dims provided: diffuse ONLY in those dimensions, fix others
        3. Generate DLA random walk with natural spreading behavior  
        4. Result: branches from same node are orthogonal by construction
        
        Parameters:
        -----------
        start_pos : np.array
            Starting position for the branch (from_node position) - shape (n_dim,)
        length : int
            Number of samples to generate for this edge
        edge_id : int
            Edge identifier for debugging/logging
        allowed_dims : np.array, optional
            Feature indices this branch is allowed to vary in. If provided,
            all other dimensions are kept fixed at start_pos values.
            Shape: (subspace_size,) containing feature indices
            
        Returns:
        --------
        np.array : Branch data of shape (length, n_dim)
        
        Example:
        --------
        Node N2 with 3 outgoing edges, n_dim=100:
        - Edge E1: allowed_dims=[0,1,2,...,33], fixes dims [34:100]
        - Edge E2: allowed_dims=[34,35,...,66], fixes dims [0:34,67:100]  
        - Edge E6: allowed_dims=[67,68,...,99], fixes dims [0:67]
        
        This ensures E1, E2, E6 can never overlap in feature space.
        """
        if length == 0:
            return np.empty((0, self.n_dim))
        
        if length == 1:
            return start_pos.reshape(1, -1)
        
        # Initialize branch with start position repeated for all samples
        branch = np.tile(start_pos, (length, 1))  # Shape: (length, n_dim)

        
        if allowed_dims is not None:
            # Constrained diffusion: only vary in allowed dimensions
            n_allowed = len(allowed_dims)


            # Generate DLA random walk only in the allowed subspace
            # First sample stays at start_pos (no displacement)
            random_increments = -0.5 + self.rand_multiplier * np.random.rand(length, n_allowed)
            random_increments[0] = 0  # First sample stays exactly at start_pos
            subspace_steps = np.cumsum(random_increments, axis=0)


            # Scale the walk appropriately
            subspace_steps = subspace_steps * 0.3

            # Apply diffusion only to allowed dimensions, others stay fixed
            branch[:, allowed_dims] += subspace_steps
            
            logging.debug(f"Constrained diffusion for edge {edge_id}: {n_allowed}/{self.n_dim} dims free")
            
        else:
            # Unconstrained diffusion: use all dimensions
            # First sample stays at start_pos (no displacement)
            random_increments = -0.5 + self.rand_multiplier * np.random.rand(length, self.n_dim)
            random_increments[0] = 0  # First sample stays exactly at start_pos
            random_steps = np.cumsum(random_increments, axis=0)


            # Scale the walk
            random_steps = random_steps * 0.3

            # Apply to all dimensions
            branch += random_steps
            
            logging.debug(f"Unconstrained diffusion for edge {edge_id}: all {self.n_dim} dims free")
        
        return branch

    def get_gt_dists(self, include_gaps: bool = False):
        """
        Compute geodesic distances as shortest paths over the complete tree graph.

        Args:
            include_gaps: If True, return complete matrix including gap samples.
                          If False, return matrix aligned to dataset.data rows only.
        """
        geodesic_dist_complete = shortest_path(
            csgraph=self.adj_matrix_complete.tocsr(), directed=False
        )

        if include_gaps:
            return geodesic_dist_complete

        # Use visible_indices to extract distances for only the visible samples
        # This aligns perfectly with dataset.data row order
        return geodesic_dist_complete[np.ix_(self.visible_indices, self.visible_indices)]

    def get_graph(self):
        """Return the precomputed adjacency graph built during data generation."""
        return self.graph


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    data_name = "gaussian_blobs" # "swiss_roll" or "saddle_surface" or "dla_tree" or "torus" or "gaussian_blobs"

    if data_name == "swiss_roll":
        dataset = SwissRoll(n_distributions=10, n_points_per_distribution=50, width=10.0, noise=0.05, manifold_noise=0.05, random_state=42, rotate_to_dim=5)

    elif data_name == "saddle_surface":
        dataset = SaddleSurface(n_distributions=10, n_points_per_distribution=50, noise=0.05, manifold_noise=0.2, a=1.0, b=1.0, random_state=42, rotate_to_dim=5)
    elif data_name == "dla_tree":
        dataset = DLAtree(n_dim=5, n_branch=5, branch_lengths=100, rand_multiplier=1.0, gap_multiplier=0.0, random_state=37, sigma=0.0, disconnect_branches=[], sampling_density_factors=None)
    elif data_name == "torus":
        dataset = Torus(n_points=500, noise=0.05, major_radius=3.0, minor_radius=1.0, random_state=42, rotate_to_dim=5)
    elif data_name == "gaussian_blobs":
        dataset = GaussianBlobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, random_state=42)
    
    data = dataset.data
    labels = dataset.metadata
    gt_distance = dataset.get_gt_dists()
    g = dataset.get_graph()
    print("Data shape:", dataset.data.shape)
    print("Labels shape:", dataset.metadata.shape)
    
    if data_name == "swiss_roll" or data_name == "saddle_surface" or data_name == "torus":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='tab20')
    elif data_name == "gaussian_blobs":
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=10)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Gaussian Blobs Dataset')
    elif data_name == "dla_tree":
        # visualize by phate
        import phate
        phate_operator = phate.PHATE()
        phate_data = phate_operator.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.scatter(phate_data[:, 0], phate_data[:, 1], c=labels, cmap="tab20", s=10)

    plt.savefig(f"{data_name}.png", bbox_inches='tight') 

    
