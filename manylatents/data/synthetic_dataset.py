import os
import torch
from torch.utils.data import Dataset
import graphtools
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import special_ortho_group
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from typing import Union, List, Optional, Dict
from .precomputed_mixin import PrecomputedMixin


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

            # Create the ground truth branch first - always connects to ground truth position
            new_branch_gt = np.cumsum(
                -1 + rand_multiplier * np.random.rand(branch_lengths[i], n_dim), axis=0
            )
            new_branch_gt += M_gt[ind]  # Connect to original (pre-jump) position

            # Create the potentially disconnected branch - starts from ground truth, then jumps
            new_branch = new_branch_gt.copy()
            if i in disconnect_branches:
                # Jump from the END of the previous branch instead of connection point
                prev_branch_end_idx = branch_start_indices[i-1] + branch_lengths[i-1] - 1
                prev_branch_end = M[prev_branch_end_idx]  # End of actual (jumped) previous branch
                jump = np.random.normal(gap_multiplier, 0.1, n_dim)  # Jump offset
                
                # Apply jump from the previous branch's end position
                jump_start = prev_branch_end + jump
                branch_offset = jump_start - new_branch[0]  # Calculate offset needed
                new_branch += branch_offset  # Apply offset to entire branch

                # Check if the jump places the branch too close to another branch
                distances = np.linalg.norm(M - new_branch[0], axis=1)
                if np.min(distances) < rand_multiplier:
                    raise ValueError(f"Jump for branch {i} is too close to another branch. Adjust gap_multiplier.")

            # Keep ground truth and actual data separate - M_gt maintains original topology
            M_gt = np.concatenate([M_gt, new_branch_gt])  # Ground truth maintains tree topology
            M = np.concatenate([M, new_branch])           # Actual data includes jumps
            branch_start_indices.append(M_gt.shape[0] - branch_lengths[i])  # Use M_gt indices

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


class DLATreeFromGraph(SyntheticDataset, PrecomputedMixin):
    def __init__(
        self,
        graph_edges,  # List of (from_node, to_node, edge_type, edge_id, length)
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
        # Deprecated parameters (kept for backwards compatibility)
        gap_edges=None,
    ):
        """
        Generate a DLA tree from explicit graph topology using pure graph traversal.
        
        Simplified approach with optional edge exclusion for gap functionality.
        
        Parameters:
        -----------
        graph_edges : list of tuples
            List of (from_node, to_node, edge_type, edge_id, length)
            - from_node, to_node: node identifiers (strings or integers)
            - edge_type: edge type for visualization (e.g., "trunk", "branch")  
            - edge_id: integer identifier for the edge (becomes population label)
            - length: number of samples to generate for this population/edge
            
        excluded_edges : list, optional
            List of edge_ids to exclude from data generation (creates "gaps").
            These edges are skipped during data generation but kept in visualization.
            
        Algorithm:
        ----------
        1. Traverse graph edges in order
        2. For each edge not in excluded_edges, generate 'length' samples using DLA random walk
        3. Each included edge represents a distinct population with edge_id as label
        4. Excluded edges create natural gaps in the data
            
        Example:
        --------
        graph_edges = [
            (1, 2, "trunk", 1, 300),    # Edge 1: 300 samples, population label=1
            (2, 3, "trunk", 2, 300),    # Edge 2: 300 samples, population label=2
            (2, 4, "branch", 3, 150),   # Edge 3: 150 samples, population label=3 
        ]
        excluded_edges = [2]  # Edge 2 will be excluded, creating a gap
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
        
        # Backward compatibility warning
        if gap_edges is not None:
            print("Warning: gap_edges parameter is deprecated and will be ignored in simplified DLA tree implementation")
        
        # Generate the data using simplified approach
        data, data_gt, metadata = self._generate_from_graph_simplified()
        
        # Load precomputed or use generated data
        if precomputed_path is not None and os.path.exists(precomputed_path):
            self.data = self.load_precomputed(precomputed_path, mmap_mode)
        else:
            self.data = data
            
        self.data_gt = data_gt
        
        # Convert edge_ids to color indices directly (same logic as graph topology)
        from manylatents.utils.mappings import cmap_dla_tree
        
        def edge_id_to_color_index(edge_id):
            """Convert edge_id to color index using same logic as graph visualization."""
            if isinstance(edge_id, str):
                return hash(edge_id) % len(cmap_dla_tree) + 1
            else:
                return edge_id  # Direct mapping: edge_id 1 → color_idx 1
        
        # Store color indices directly as metadata
        self.metadata = np.array([edge_id_to_color_index(label) for label in metadata])
        
        # Generate and save visualization
        if save_graph_viz:
            self._visualize_and_save_graph()

    def _generate_from_graph(self):
        """Generate DLA tree from graph specification."""
        np.random.seed(self.random_state)
        
        # Step 1: Determine unique nodes and create positions
        all_nodes = set()
        for from_node, to_node, _, _, _ in self.graph_edges:
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        for from_node, to_node, _ in self.gap_edges:
            all_nodes.add(from_node) 
            all_nodes.add(to_node)
        
        # Step 2: Assign random positions to nodes in high-dimensional space
        node_positions = {}
        for i, node in enumerate(sorted(all_nodes)):
            if i == 0:  # Root node at origin
                node_positions[node] = np.zeros(self.n_dim)
            else:
                # Random position for this node
                node_positions[node] = np.random.randn(self.n_dim) * 10
        
        # Step 3: Generate DLA branches for each edge
        M = np.empty((0, self.n_dim))
        M_gt = np.empty((0, self.n_dim))
        metadata_list = []
        
        # Create set of gap edge pairs for quick lookup
        gap_edge_pairs = {(from_node, to_node) for from_node, to_node, _ in self.gap_edges}
        
        for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
            # Skip edges that are defined as gaps - they shouldn't have data
            if (from_node, to_node) in gap_edge_pairs:
                continue
                
            # Generate DLA branch from from_node to to_node for data edges only
            branch_data = self._generate_dla_branch(
                start_pos=node_positions[from_node],
                end_pos=node_positions[to_node], 
                length=length,
                branch_id=edge_id
            )
            
            M = np.concatenate([M, branch_data])
            M_gt = np.concatenate([M_gt, branch_data])  # Same as M for now
            metadata_list.extend([edge_id] * length)
        
        # Step 4: Apply gaps by moving nodes connected by gap_edges
        for from_node, to_node, gap_size in self.gap_edges:
            gap_vector = np.random.randn(self.n_dim) * gap_size
            node_positions[to_node] += gap_vector
            
            # Update all branches that start from to_node
            current_idx = 0
            for from_node_edge, to_node_edge, edge_type, edge_id, length in self.graph_edges:
                if from_node_edge == to_node:
                    # This branch starts from the moved node, update its position
                    M[current_idx:current_idx + length] += gap_vector
                current_idx += length
        
        # Step 5: Add noise
        noise = np.random.normal(0, self.sigma, M.shape)
        M += noise
        M_gt += noise
        
        metadata = np.array(metadata_list)
        return M, M_gt, metadata

    def _generate_from_graph_simplified(self):
        """
        Simplified DLA tree generation using pure graph traversal.
        
        Algorithm:
        1. Build graph connectivity from edges  
        2. Traverse graph starting from root
        3. For each edge, generate DLA random walk with specified length
        4. Each edge becomes a population with edge_id as label
        """
        np.random.seed(self.random_state)
        
        # Build adjacency structure and find root
        graph_dict = {}  # node -> list of (target_node, edge_type, edge_id, length)
        all_nodes = set()
        
        for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
            if from_node not in graph_dict:
                graph_dict[from_node] = []
            graph_dict[from_node].append((to_node, edge_type, edge_id, length))
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        
        # Find root node (node that appears in from_node but never in to_node)
        to_nodes = {to_node for from_node, to_node, _, _, _ in self.graph_edges}
        root_candidates = [node for node in all_nodes if node not in to_nodes]
        
        if not root_candidates:
            # If no clear root, use the first node
            root_node = sorted(all_nodes)[0]
        else:
            root_node = root_candidates[0]
        
        print(f"Starting DLA tree generation from root node: {root_node}")
        
        # Initialize data storage
        all_data = []
        all_metadata = []
        node_positions = {root_node: np.zeros(self.n_dim)}  # Root at origin
        
        # Use iterative approach to avoid recursion issues with cycles
        # Simply generate all edges in the order they appear in graph_edges
        for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
            print(f"Generating edge {edge_id}: {from_node} -> {to_node} ({length} samples)")
            
            # Get starting position (from_node position or origin if not set)
            if from_node in node_positions:
                start_position = node_positions[from_node]
            else:
                # If from_node not positioned yet, place it at origin + small offset
                start_position = np.random.randn(self.n_dim) * 0.5
                node_positions[from_node] = start_position
            
            # Generate DLA branch for this edge
            branch_data = self._generate_dla_branch_simple(
                start_pos=start_position,
                length=length
            )
            
            # Always add to data for proper connectivity
            all_data.append(branch_data)
            all_metadata.extend([edge_id] * length)
            
            # Set target position as end of this branch
            target_position = branch_data[-1]
            node_positions[to_node] = target_position
        
        # Combine all branch data
        if all_data:
            M = np.vstack(all_data)
        else:
            M = np.empty((0, self.n_dim))
            
        M_gt = M.copy()  # Ground truth same as generated data
        metadata = np.array(all_metadata)
        
        # Apply gap functionality: remove excluded edges AFTER generating all data
        if self.excluded_edges:
            # Create mask for samples we want to KEEP (not in excluded edges)
            keep_mask = ~np.isin(metadata, list(self.excluded_edges))
            
            # Apply logical subsetting
            M = M[keep_mask]
            M_gt = M_gt[keep_mask]  
            metadata = metadata[keep_mask]
            
            excluded_edges_found = set(all_metadata) & self.excluded_edges
            print(f"Applied gaps: removed edges {sorted(excluded_edges_found)} from data")
            print(f"Remaining edges: {sorted(set(metadata))}")
            
            # Renumber visible edges to be sequential 1, 2, 3, ..., n_visible_edges
            unique_visible_edges = sorted(set(metadata))
            edge_renumbering = {old_id: new_id for new_id, old_id in enumerate(unique_visible_edges, 1)}
            
            # Apply renumbering to metadata
            renumbered_metadata = np.array([edge_renumbering[old_id] for old_id in metadata])
            
            print(f"Edge renumbering: {edge_renumbering}")
            print(f"Final visible edges: {sorted(set(renumbered_metadata))}")
            
            metadata = renumbered_metadata
            
            # Store renumbering for visualization
            self.edge_renumbering = edge_renumbering
            self.original_excluded_edges = excluded_edges_found
        else:
            self.edge_renumbering = None
            self.original_excluded_edges = set()
        
        # Add global noise
        if self.sigma > 0:
            noise = np.random.normal(0, self.sigma, M.shape)
            M += noise
            M_gt += noise
        
        print(f"Final DLA tree: {len(M)} total samples across {len(set(metadata)) if len(metadata) > 0 else 0} populations")
        
        return M, M_gt, metadata

    def _generate_dla_branch_simple(self, start_pos, length):
        """
        Generate a simple DLA branch using cumulative random walk.
        
        Based on PHATE tree.py approach - generates natural branching patterns.
        """
        if length == 0:
            return np.empty((0, self.n_dim))
        
        if length == 1:
            return start_pos.reshape(1, -1)
        
        # Generate cumulative random walk
        # Use random steps scaled by rand_multiplier
        random_steps = np.cumsum(
            -1 + self.rand_multiplier * np.random.rand(length, self.n_dim), 
            axis=0
        )
        
        # Scale the walk to have reasonable magnitude
        # Each step should be small relative to overall tree size
        random_steps = random_steps * 0.5  # Scale factor for tree size
        
        # Add start position to get absolute positions
        branch = start_pos + random_steps
        
        return branch

    def _create_hierarchical_layout(self, G):
        """Create a custom layout that avoids edge overlaps by using hierarchical positioning."""
        import networkx as nx
        import numpy as np
        
        # Find root node (no incoming edges in a DAG, or use degree-based heuristic)
        all_nodes = set(G.nodes())
        target_nodes = set()
        for u, v in G.edges():
            target_nodes.add(v)
        
        root_candidates = all_nodes - target_nodes
        if root_candidates:
            root = min(root_candidates)  # Use smallest node ID as root
        else:
            # If cyclic, use node with highest out-degree
            root = max(G.nodes(), key=lambda n: G.out_degree(n))
        
        # Build levels using BFS from root
        levels = {}
        queue = [(root, 0)]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            # Add neighbors to next level
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))
        
        # Position nodes hierarchically
        pos = {}
        max_level = max(levels.keys()) if levels else 0
        
        for level, nodes in levels.items():
            n_nodes = len(nodes)
            if n_nodes == 1:
                # Single node: center it
                pos[nodes[0]] = (0, max_level - level)
            else:
                # Multiple nodes: spread them horizontally
                x_positions = np.linspace(-n_nodes/2, n_nodes/2, n_nodes)
                for i, node in enumerate(sorted(nodes)):
                    pos[node] = (x_positions[i], max_level - level)
        
        # Handle any remaining nodes (in case of complex connectivity)
        for node in G.nodes():
            if node not in pos:
                pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-1, 1))
        
        return pos

    def _create_better_layout(self, G):
        """Try multiple layout algorithms to find one that looks good."""
        import networkx as nx
        import numpy as np
        
        # Find root node for tree-based layouts
        all_nodes = set(G.nodes())
        target_nodes = set(v for u, v in G.edges())
        root_candidates = all_nodes - target_nodes
        root = min(root_candidates) if root_candidates else min(G.nodes())
        
        try:
            # Try graphviz dot layout (hierarchical, clean)
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            print("Using graphviz dot layout")
            return pos
        except:
            pass
        
        try:
            # Try networkx tree layout
            pos = nx.nx_agraph.graphviz_layout(G, prog='neato')  
            print("Using graphviz neato layout")
            return pos
        except:
            pass
        
        try:
            # Try kamada-kawai layout (force-directed but more stable)
            pos = nx.kamada_kawai_layout(G, scale=2)
            print("Using Kamada-Kawai layout")
            return pos
        except:
            pass
        
        # Fallback: improved spring layout with better parameters
        print("Using improved spring layout")
        pos = nx.spring_layout(
            G, 
            k=3,  # Increase spacing between nodes
            iterations=100,  # More iterations for better convergence
            scale=2,  # Larger scale
            seed=self.random_state
        )
        
        return pos
    
    def _create_semantic_layout(self, G):
        """
        Create a layout based purely on graph structure without automatic layout algorithms.
        Positions nodes based on their semantic relationships in the graph.
        """
        import numpy as np
        import networkx as nx
        
        # Find the root node (node that appears as source but not as target)
        all_nodes = set(G.nodes())
        target_nodes = {v for u, v in G.edges()}
        root_candidates = all_nodes - target_nodes
        root = min(root_candidates) if root_candidates else min(G.nodes())
        
        pos = {}
        
        # Build tree levels using BFS traversal
        levels = {0: [root]}
        visited = {root}
        queue = [(root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            # Add children to next level
            children = [neighbor for neighbor in G.neighbors(node) if neighbor not in visited]
            if children:
                next_level = level + 1
                if next_level not in levels:
                    levels[next_level] = []
                for child in children:
                    levels[next_level].append(child)
                    visited.add(child)
                    queue.append((child, next_level))
        
        # Position nodes semantically
        # Root at origin, each level gets positioned based on graph structure
        pos[root] = (0, 0)
        
        for level_num, nodes in levels.items():
            if level_num == 0:  # Root level already positioned
                continue
                
            y_pos = -level_num * 2  # Move down for each level
            
            # For each node in this level, position based on its parent
            for i, node in enumerate(nodes):
                # Find parent (node that connects to this one from previous level)
                parent = None
                for prev_level_node in levels.get(level_num - 1, []):
                    if G.has_edge(prev_level_node, node):
                        parent = prev_level_node
                        break
                
                if parent and parent in pos:
                    # Position relative to parent
                    parent_x = pos[parent][0]
                    # Spread children horizontally around parent
                    parent_children = [n for n in nodes if any(G.has_edge(p, n) for p in levels.get(level_num - 1, []) if p == parent)]
                    child_index = parent_children.index(node)
                    n_children = len(parent_children)
                    
                    if n_children == 1:
                        x_pos = parent_x
                    else:
                        # Spread children around parent
                        spacing = 3.0  # Horizontal spacing between siblings
                        x_offset = (child_index - (n_children - 1) / 2) * spacing
                        x_pos = parent_x + x_offset
                else:
                    # Fallback: spread nodes horizontally at this level
                    spacing = 4.0
                    x_pos = (i - (len(nodes) - 1) / 2) * spacing
                
                pos[node] = (x_pos, y_pos)
        
        # Handle any unpositioned nodes (shouldn't happen with well-formed trees)
        for node in G.nodes():
            if node not in pos:
                pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
        
        return pos

    def _create_proportional_layout(self, G):
        """Create a layout where edge lengths are proportional to sample counts."""
        import numpy as np
        import networkx as nx
        
        # First, create a good-looking layout using spring layout
        pos = nx.spring_layout(G, seed=self.random_state, k=2, iterations=50)
        
        # Now adjust edge lengths to be proportional to sample counts
        # We'll do this iteratively by moving nodes to achieve target edge lengths
        target_lengths = {}
        
        # Calculate target lengths for each edge based on sample counts
        for u, v, d in G.edges(data=True):
            if 'gap_size' not in d:  # Only for data edges
                sample_count = d.get('length', 300)
                # Normalize to reasonable visual length (baseline 300 samples = 1.0 unit)
                target_lengths[(u, v)] = sample_count / 300.0
        
        # Iteratively adjust positions to match target edge lengths
        for iteration in range(20):  # Limited iterations to avoid infinite loops
            for (u, v), target_length in target_lengths.items():
                if u in pos and v in pos:
                    # Current edge vector
                    edge_vec = pos[v] - pos[u]
                    current_length = np.linalg.norm(edge_vec)
                    
                    if current_length > 0:  # Avoid division by zero
                        # Scale edge to target length
                        direction = edge_vec / current_length
                        new_edge_vec = direction * target_length
                        
                        # Move the second node to achieve target length
                        # (keep first node fixed for stability)
                        pos[v] = pos[u] + new_edge_vec
        
        return pos
    
    def _generate_dla_branch(self, start_pos, end_pos, length, branch_id):
        """Generate a DLA branch between two positions."""
        # Direction vector from start to end
        direction = end_pos - start_pos
        
        # Generate DLA-style cumulative random walk
        random_steps = np.cumsum(
            -1 + self.rand_multiplier * np.random.rand(length, self.n_dim), axis=0
        )
        
        # Scale and orient the random walk to go from start to end
        if length > 1:
            # Normalize random walk to unit length, then scale to desired direction
            random_steps = random_steps / np.linalg.norm(random_steps[-1]) * np.linalg.norm(direction)
            random_steps = random_steps + direction * np.linspace(0, 1, length).reshape(-1, 1)
        
        # Add start position
        branch = start_pos + random_steps
        return branch
        
    def _visualize_and_save_graph(self):
        """Create and save a visualization of the graph topology."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create networkx graph
            G = nx.DiGraph()
            
            # Add all edges (simplified - no gaps to skip)
            for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
                G.add_edge(from_node, to_node, edge_type=edge_type, edge_id=edge_id, length=length)
            
            # Use custom semantic layout based on graph structure
            pos = self._create_semantic_layout(G)
            
            # Plot
            plt.figure(figsize=(6, 4))
            
            # Don't draw nodes - they're just placeholders
            # The meaningful data clusters are the EDGES (branches)
            
            # Get colormap for edge coloring (same as embeddings)
            from manylatents.utils.mappings import cmap_dla_tree
            
            # All edges are data edges (simplified approach - no gaps)
            regular_edges = [(u, v, d) for u, v, d in G.edges(data=True)]
            
            # Separate edges into visible and gap edges for different drawing styles
            visible_edges = []
            gap_edges = []
            visible_edge_colors = []
            visible_edge_labels = {}
            gap_edge_labels = {}
            
            for u, v, d in regular_edges:
                original_edge_id = d['edge_id']
                
                if original_edge_id in self.original_excluded_edges:
                    # Gap edge: will be drawn as transparent outline
                    gap_edges.append((u, v))
                    gap_edge_labels[(u, v)] = f"({original_edge_id})"  # Parentheses to indicate gap
                else:
                    # Visible edge: use renumbered ID for coloring and labeling
                    visible_edges.append((u, v))
                    
                    if self.edge_renumbering is not None:
                        display_edge_id = self.edge_renumbering.get(original_edge_id, original_edge_id)
                    else:
                        display_edge_id = original_edge_id
                    
                    # Map display_edge_id to color index
                    if isinstance(display_edge_id, str):
                        color_idx = hash(display_edge_id) % len(cmap_dla_tree) + 1
                    else:
                        color_idx = display_edge_id
                    
                    color = cmap_dla_tree[color_idx]
                    visible_edge_colors.append(color)
                    visible_edge_labels[(u, v)] = f"{display_edge_id}"
            
            # Draw visible edges (data branches) with colormap colors
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges, 
                                     edge_color=visible_edge_colors, width=6, alpha=0.8, arrows=False)
            
            # Draw gap edges as transparent outlines
            if gap_edges:
                nx.draw_networkx_edges(G, pos, edgelist=gap_edges, 
                                     edge_color='gray', width=3, alpha=0.3, arrows=False, 
                                     style='dashed')
            
            # Add edge labels for visible edges
            if visible_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, visible_edge_labels, font_size=8)
            
            # Add edge labels for gap edges (in parentheses to indicate they're gaps)
            if gap_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, gap_edge_labels, font_size=7, font_color='gray', alpha=0.7)
            
            # Dynamic title based on whether gaps are present
            if self.original_excluded_edges:
                gap_info = f"(Dashed = Gaps: {sorted(self.original_excluded_edges)})"
                plt.title(f"DLA Tree Graph Topology\n{gap_info}")
            else:
                plt.title("DLA Tree Graph Topology\n(All Edges Visible)")
            plt.axis('off')
            
            # Save
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, "dla_tree_graph_topology.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Graph topology visualization saved to: {save_path}")
            
        except ImportError:
            print("NetworkX or matplotlib not available for graph visualization")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    data_name = "dla_tree" # "swiss_roll" or "saddle_surface" or "dla_tree"

    if data_name == "swiss_roll":
        dataset = SwissRoll(n_distributions=10, n_points_per_distribution=50, width=10.0, noise=0.05, manifold_noise=0.05, random_state=42, rotate_to_dim=5)

    elif data_name == "saddle_surface":
        dataset = SaddleSurface(n_distributions=10, n_points_per_distribution=50, noise=0.05, manifold_noise=0.2, a=1.0, b=1.0, random_state=42, rotate_to_dim=5)
    elif data_name == "dla_tree":
        dataset = DLAtree(n_dim=5, n_branch=5, branch_lengths=100, rand_multiplier=1.0, gap_multiplier=0.0, random_state=37, sigma=0.0, disconnect_branches=[], sampling_density_factors=None)
    
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

    
