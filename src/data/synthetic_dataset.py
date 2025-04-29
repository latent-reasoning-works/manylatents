import torch
from torch.utils.data import Dataset
import graphtools
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import special_ortho_group
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from typing import Union, List, Optional, Dict


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

    #@property
    #def geodesic_dists(self):
    #    D = self.get_geodesic()
    #    return D[np.triu_indices(D.shape[0], k=1)]


class SwissRoll(SyntheticDataset):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.1,
        manifold_noise=0.1,
        width=10.0,
        random_state=42,
        rotate_to_dim=3,
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
        """
        super().__init__()
        rng = np.random.default_rng(random_state)

        self.mean_t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, n_distributions))
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
        X += noise * rng.normal(size=(3, n_distributions * n_points_per_distribution))
        self.data = X.T  # shape (5000, 3)
        self.ts = np.squeeze(ts)  # (5000,)
        self.metadata = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        self.t = self.mean_t[0]  # shape (100, )
        mean_x = self.mean_t * np.cos(self.mean_t)  # shape (1, 100)
        mean_z = self.mean_t * np.sin(self.mean_t)  # shape (1, 100)
        self.means = np.concatenate((mean_x, self.mean_y, mean_z)).T  # shape (100, 3)
        if rotate_to_dim > 3:
            self.data = self.rotate_to_dim(rotate_to_dim)
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
        # u_t = self.ts
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


class DLAtree(SyntheticDataset):
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

        self.data = data  # noisy data
        self.data_gt = data_gt  # ground truth tree
        self.metadata = metadata
        self.n_dim = n_dim
        
    def get_geodesic(self):
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



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    data_name = "dla_tree" # "swiss_roll" or "saddle_surface" or "dla_tree"

    if data_name == "swiss_roll":
        dataset = SwissRoll(n_distributions=100, n_points_per_distribution=50, width=10.0, noise=0.05, manifold_noise=0.05, random_state=42, rotate_to_dim=5)

    elif data_name == "saddle_surface":
        dataset = SaddleSurface(n_distributions=100, n_points_per_distribution=50, noise=0.05, manifold_noise=0.2, a=1.0, b=1.0, random_state=42, rotate_to_dim=5)
    elif data_name == "dla_tree":
        dataset = DLAtree(n_dim=100, n_branch=20, branch_lengths=100, rand_multiplier=2.0, gap_multiplier=0.0, random_state=37, sigma=4.0, disconnect_branches=[], sampling_density_factors=None)
    
    data = dataset.data
    labels = dataset.metadata
    gt_distance = dataset.get_gt_dists()
    g = dataset.get_graph()
    print("Data shape:", dataset.data.shape)
    print("Labels shape:", dataset.metadata.shape)
    
    if data_name == "swiss_roll" or data_name == "saddle_surface":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,2], data[:,0], data[:,1], c=labels, cmap='tab20')
    elif data_name == "dla_tree":
        # visualize by phate
        import phate
        phate_operator = phate.PHATE()
        phate_data = phate_operator.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.scatter(phate_data[:, 0], phate_data[:, 1], c=labels, cmap="tab20", s=10)

    plt.savefig(f"{data_name}.png", bbox_inches='tight') 

    
