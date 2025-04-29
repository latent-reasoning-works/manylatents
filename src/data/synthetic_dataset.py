import torch
from torch.utils.data import Dataset
import graphtools
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import special_ortho_group


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
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    data_name = "saddle_surface" # "swiss_roll"

    if data_name == "swiss_roll":
        dataset = SwissRoll(n_distributions=100, n_points_per_distribution=50, width=10.0, noise=0.05, manifold_noise=0.05, random_state=42, rotate_to_dim=5)

    elif data_name == "saddle_surface":
        dataset = SaddleSurface(n_distributions=100, n_points_per_distribution=50, noise=0.05, manifold_noise=0.2, a=1.0, b=1.0, random_state=42, rotate_to_dim=5)
    
    data = dataset.X
    labels = dataset.metadata
    gt_distance = dataset.get_gt_dists()
    print("Data shape:", dataset.X.shape)
    print("Labels shape:", dataset.labels.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='tab20')
    plt.savefig(f"{data_name}.png", bbox_inches='tight') 

    
