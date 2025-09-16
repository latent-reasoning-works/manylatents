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
             for i in [branch_idx + 1] * branch_len]  # Add 1 to start labels from 1 instead of 0
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
            List of (from_node, to_node, edge_type, edge_id, length) where:
            - from_node, to_node: Node identifiers (strings or integers)
            - edge_type: Edge classification ("trunk", "branch", "cross") for visualization
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
        
        gap_edges : deprecated
            Legacy parameter, use excluded_edges instead.
            
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
            (1, 2, "trunk", 1, 300),    # Main trunk: 300 samples, edge 1
            (2, 3, "trunk", 2, 300),    # Continuation: 300 samples, edge 2  
            (2, 4, "branch", 3, 150),   # Side branch: 150 samples, edge 3
            (2, 5, "cross", 4, 75),     # Cross-link: 75 samples, edge 4
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
        
        # Backward compatibility warning
        if gap_edges is not None:
            print("Warning: gap_edges parameter is deprecated and will be ignored in simplified DLA tree implementation")
        
        # Generate the data using simplified approach
        data, data_gt, metadata = self._generate_from_graph()
        
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
        """
        Generate DLA tree that properly follows the graph topology.
        
        Algorithm:
        1. Build graph connectivity from edges and establish node positions
        2. For each edge [from_node, to_node, edge_type, edge_id, length]:
           - Start DLA random walk at from_node's position
           - Generate 'length' samples via random walk  
           - End walk at to_node's position (enforced connectivity)
           - Assign all samples the edge_id label
        3. Result: DLA tree structure matches graph topology exactly
        """
        np.random.seed(self.random_state)
        
        # Step 1: Identify all unique nodes and find root
        all_nodes = set()
        for from_node, to_node, _, _, _ in self.graph_edges:
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        
        # Find root node (appears as from_node but never as to_node)
        to_nodes = {to_node for from_node, to_node, _, _, _ in self.graph_edges}
        root_candidates = [node for node in all_nodes if node not in to_nodes]
        root_node = root_candidates[0] if root_candidates else sorted(all_nodes)[0]
        
        print(f"Starting DLA tree generation from root node: {root_node}")
        print(f"Total nodes in graph: {sorted(all_nodes)}")
        
        # Step 2: Establish node positions by processing edges in dependency order
        node_positions = {}
        all_data = []
        all_metadata = []
        
        # Place root at origin
        node_positions[root_node] = np.zeros(self.n_dim)
        print(f"Root node {root_node} positioned at origin")
        
        # Step 2.5: Analyze node degrees and create orthogonal direction vectors
        node_outgoing_edges = {}  # node -> list of (to_node, edge_id)
        for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
            if from_node not in node_outgoing_edges:
                node_outgoing_edges[from_node] = []
            node_outgoing_edges[from_node].append((to_node, edge_id))
        
        # Create orthogonal subspaces for nodes with multiple outgoing edges
        node_subspaces = {}
        for node, outgoing_edges in node_outgoing_edges.items():
            if len(outgoing_edges) > 1:
                print(f"Node {node} has {len(outgoing_edges)} outgoing edges: {[edge_id for _, edge_id in outgoing_edges]}")
                node_subspaces[node] = self._create_orthogonal_subspaces(
                    n_directions=len(outgoing_edges),
                    n_dim=self.n_dim,
                    node_id=node
                )
        
        # Process edges to establish node positions and generate data
        processed_edges = set()
        max_iterations = len(self.graph_edges) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(processed_edges) < len(self.graph_edges) and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
                # Skip if already processed
                if (from_node, to_node, edge_id) in processed_edges:
                    continue
                    
                # Can only process if from_node has a known position
                if from_node not in node_positions:
                    continue
                    
                print(f"Processing edge {edge_id}: {from_node} -> {to_node} ({length} samples)")
                
                # Get starting position
                start_pos = node_positions[from_node]
                
                # Get orthogonal subspace if this node has multiple outgoing edges
                allowed_dims = None
                if from_node in node_subspaces:
                    # Find which subspace corresponds to this edge_id
                    outgoing_edges = node_outgoing_edges[from_node]
                    for i, (_, out_edge_id) in enumerate(outgoing_edges):
                        if out_edge_id == edge_id:
                            allowed_dims = node_subspaces[from_node][i]
                            print(f"  Using subspace {i} for edge {edge_id} from node {from_node}: dims {allowed_dims[:3]}...{allowed_dims[-3:]}")
                            break
                
                # Generate DLA branch from start to target
                branch_data = self._generate_dla_branch_topology_aware(
                    start_pos=start_pos,
                    length=length,
                    edge_id=edge_id,
                    allowed_dims=allowed_dims
                )
                
                # Store data and metadata
                all_data.append(branch_data)
                all_metadata.extend([edge_id] * length)
                
                # Set target node position as the end of this branch
                node_positions[to_node] = branch_data[-1].copy()
                print(f"Node {to_node} positioned at end of edge {edge_id}")
                
                # Mark as processed
                processed_edges.add((from_node, to_node, edge_id))
                made_progress = True
            
            if not made_progress:
                # Handle remaining edges by placing missing nodes randomly
                for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
                    if (from_node, to_node, edge_id) in processed_edges:
                        continue
                        
                    print(f"Handling unconnected edge {edge_id}: {from_node} -> {to_node}")
                    
                    # Place from_node randomly if not positioned
                    if from_node not in node_positions:
                        node_positions[from_node] = np.random.randn(self.n_dim) * 2.0
                        print(f"Randomly positioned unconnected node {from_node}")
                    
                    # Generate branch (with potential orthogonal subspace)
                    start_pos = node_positions[from_node]
                    allowed_dims = None
                    if from_node in node_subspaces:
                        outgoing_edges = node_outgoing_edges[from_node]
                        for i, (_, out_edge_id) in enumerate(outgoing_edges):
                            if out_edge_id == edge_id:
                                allowed_dims = node_subspaces[from_node][i]
                                break
                    
                    branch_data = self._generate_dla_branch_topology_aware(
                        start_pos=start_pos,
                        length=length,
                        edge_id=edge_id,
                        allowed_dims=allowed_dims
                    )
                    
                    all_data.append(branch_data)
                    all_metadata.extend([edge_id] * length)
                    node_positions[to_node] = branch_data[-1].copy()
                    processed_edges.add((from_node, to_node, edge_id))
        
        print(f"Processed {len(processed_edges)} edges in {iteration} iterations")
        print(f"Final node positions: {list(node_positions.keys())}")
        
        # Step 3: Combine all branch data
        if all_data:
            M = np.vstack(all_data)
        else:
            M = np.empty((0, self.n_dim))
            
        M_gt = M.copy()  # Ground truth same as generated data
        metadata = np.array(all_metadata)
        
        # Step 4: Apply gap functionality (remove excluded edges)
        if self.excluded_edges:
            keep_mask = ~np.isin(metadata, list(self.excluded_edges))
            M = M[keep_mask]
            M_gt = M_gt[keep_mask]  
            metadata = metadata[keep_mask]
            
            excluded_edges_found = set(all_metadata) & self.excluded_edges
            print(f"Applied gaps: removed edges {sorted(excluded_edges_found)} from data")
            print(f"Remaining edges: {sorted(set(metadata))}")
            
            # Renumber visible edges to be sequential
            unique_visible_edges = sorted(set(metadata))
            edge_renumbering = {old_id: new_id for new_id, old_id in enumerate(unique_visible_edges, 1)}
            renumbered_metadata = np.array([edge_renumbering[old_id] for old_id in metadata])
            
            print(f"Edge renumbering: {edge_renumbering}")
            metadata = renumbered_metadata
            
            self.edge_renumbering = edge_renumbering
            self.original_excluded_edges = excluded_edges_found
        else:
            self.edge_renumbering = None
            self.original_excluded_edges = set()
        
        # Step 5: Add global noise
        if self.sigma > 0:
            noise = np.random.normal(0, self.sigma, M.shape)
            M += noise
            M_gt += noise
        
        print(f"Final DLA tree: {len(M)} total samples across {len(set(metadata)) if len(metadata) > 0 else 0} populations")
        
        return M, M_gt, metadata

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
        
        print(f"  Creating {n_directions} orthogonal subspaces for node {node_id}")
        print(f"  Each subspace ≈ {subspace_size} dims, {remaining_dims} dims distributed")
        
        subspaces = []
        current_start = 0
        
        for i in range(n_directions):
            # Give extra dimensions to first few branches if needed
            current_size = subspace_size + (1 if i < remaining_dims else 0)
            current_end = current_start + current_size
            
            # Assign this range of features to this branch
            subspace_indices = np.arange(current_start, current_end)
            subspaces.append(subspace_indices)
            
            print(f"    Branch {i}: diffuses in features [{current_start}:{current_end}] ({current_size} dims)")
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
            subspace_steps = np.cumsum(
                -0.5 + self.rand_multiplier * np.random.rand(length, n_allowed), 
                axis=0
            )
            
            # Scale the walk appropriately
            subspace_steps = subspace_steps * 0.3
            
            # Apply diffusion only to allowed dimensions, others stay fixed
            branch[:, allowed_dims] += subspace_steps
            
            print(f"    Constrained diffusion for edge {edge_id}: {n_allowed}/{self.n_dim} dims free")
            
        else:
            # Unconstrained diffusion: use all dimensions
            random_steps = np.cumsum(
                -0.5 + self.rand_multiplier * np.random.rand(length, self.n_dim), 
                axis=0
            )
            
            # Scale the walk
            random_steps = random_steps * 0.3
            
            # Apply to all dimensions
            branch += random_steps
            
            print(f"    Unconstrained diffusion for edge {edge_id}: all {self.n_dim} dims free")
        
        return branch

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
        """Create and save two versions of graph topology visualization."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import os
            
            # Create networkx graph
            G = nx.DiGraph()
            
            # Add all edges
            for from_node, to_node, edge_type, edge_id, length in self.graph_edges:
                G.add_edge(from_node, to_node, edge_type=edge_type, edge_id=edge_id, length=length)
            
            # Use custom semantic layout based on graph structure
            pos = self._create_semantic_layout(G)
            
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
                    gap_edge_labels[(u, v)] = f"E{original_edge_id}"  # E prefix for gap edges
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
                    visible_edge_labels[(u, v)] = f"{display_edge_id}"  # No E prefix for display version
            
            os.makedirs(self.save_dir, exist_ok=True)
            
            # ===== Display Version (Clean, no node labels) =====
            plt.figure(figsize=(10, 8))
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
            
            # Draw visible edges (data branches) with colormap colors
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges, 
                                     edge_color=visible_edge_colors, width=8, alpha=0.9, arrows=False,
                                     arrowstyle='-', arrowsize=20)
            
            # Draw gap edges as faint dashed lines
            if gap_edges:
                nx.draw_networkx_edges(G, pos, edgelist=gap_edges, 
                                     edge_color='lightgray', width=3, alpha=0.4, arrows=False, 
                                     style='dashed', arrowstyle='-')
            
            # Add clean edge labels for visible edges only
            if visible_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, visible_edge_labels, font_size=12, 
                                           font_weight='bold', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.3', 
                                                    facecolor='white', alpha=0.9, edgecolor='none'))
            
            # Clean title for display
            plt.title("DLA Tree Graph Topology", fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Save display version
            display_path = os.path.join(self.save_dir, "dla_tree_graph_topology.png")
            plt.savefig(display_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Display graph visualization saved to: {display_path}")
            
            # ===== Debug Version (Detailed, with node and edge labels) =====
            plt.figure(figsize=(12, 10))
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
            
            # Draw nodes with labels
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.8)
            
            # Draw visible edges (data branches) with colormap colors
            if visible_edges:
                nx.draw_networkx_edges(G, pos, edgelist=visible_edges, 
                                     edge_color=visible_edge_colors, width=6, alpha=0.8, arrows=True,
                                     arrowstyle='->', arrowsize=20, connectionstyle='arc3,rad=0.1')
            
            # Draw gap edges as dashed outlines
            if gap_edges:
                nx.draw_networkx_edges(G, pos, edgelist=gap_edges, 
                                     edge_color='gray', width=4, alpha=0.5, arrows=True, 
                                     style='dashed', arrowstyle='->', arrowsize=15,
                                     connectionstyle='arc3,rad=0.1')
            
            # Add node labels (N1, N2, ...)
            node_labels = {node: f"N{node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight='bold', 
                                  font_family='sans-serif')
            
            # Add edge labels for visible edges (E1, E2, ...)
            if visible_edge_labels:
                nx.draw_networkx_edge_labels(G, pos, visible_edge_labels, font_size=10, 
                                           font_weight='bold', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.2', 
                                                    facecolor='yellow', alpha=0.7, edgecolor='black'))
            
            # Add edge labels for gap edges (E9, E10, ... in parentheses)
            if gap_edge_labels:
                gap_labels_formatted = {edge: f"({label})" for edge, label in gap_edge_labels.items()}
                nx.draw_networkx_edge_labels(G, pos, gap_labels_formatted, font_size=9, 
                                           font_color='darkgray', font_family='sans-serif',
                                           bbox=dict(boxstyle='round,pad=0.2', 
                                                    facecolor='lightgray', alpha=0.6, edgecolor='gray'))
            
            # Detailed title with gap information
            if self.original_excluded_edges:
                gap_info = f"Gaps (Excluded): {sorted(self.original_excluded_edges)}"
                plt.title(f"DLA Tree Graph Topology - DEBUG\n{gap_info}\nSolid edges = Data populations, Dashed edges = Excluded (no data)", 
                         fontsize=14, pad=20)
            else:
                plt.title("DLA Tree Graph Topology - DEBUG\nAll Edges Visible", 
                         fontsize=14, pad=20)
            plt.axis('off')
            
            # Save debug version
            debug_path = os.path.join(self.save_dir, "dla_tree_graph_topology_debug.png")
            plt.savefig(debug_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Debug graph visualization saved to: {debug_path}")
            
        except ImportError:
            print("NetworkX or matplotlib not available for graph visualization")

    def get_gt_dists(self):
        """
        Compute geodesic distances as shortest paths over the tree graph.
        Uses the same approach as DLATree.
        """
        if self.graph is None:
            self.get_graph()
        geodesic_dist = shortest_path(csgraph=self.graph.tocsr(), directed=False)
        return geodesic_dist

    def get_graph(self):
        """
        Build a sparse adjacency graph connecting neighboring points along the tree structure.
        Uses the same approach as DLATree, connecting consecutive points in data_gt.
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
