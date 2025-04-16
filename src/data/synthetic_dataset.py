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
        self.X = None
        self.labels = None
        self.graph = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.labels[idx] if self.labels is not None else -1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.X

    def standardize_data(self):
        """
        Standardize data putting it in a unit box around the origin. (min_max normalization)
        This is necessary for quadtree type algorithms
        """
        X = self.X
        minx = np.min(self.X, axis=0)
        maxx = np.max(self.X, axis=0)
        self.std_X = (X - minx) / (maxx - minx)
        return self.std_X

    def rotate_to_dim(self, dim):
        """
        Rotate dataset to a different dimensionality.
        """
        self.rot_mat = special_ortho_group.rvs(dim)[: self.X.shape[1]]
        self.high_X = np.dot(self.X, self.rot_mat)
        return self.high_X

    def get_geodesic(self):
        pass

class SwissRoll(SyntheticDataset):
    def __init__(
        self,
        n_distributions=100,
        n_points_per_distribution=50,
        noise=0.0,
        manifold_noise=1.0,
        width=1,
        random_state=42,
        rotate_to_dim=3,
    ):
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
        self.X = X.T  # shape (5000, 3)
        self.ts = np.squeeze(ts)  # (5000,)
        self.labels = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        self.t = self.mean_t[0]  # shape (100, )
        mean_x = self.mean_t * np.cos(self.mean_t)  # shape (1, 100)
        mean_z = self.mean_t * np.sin(self.mean_t)  # shape (1, 100)
        self.means = np.concatenate((mean_x, self.mean_y, mean_z)).T  # shape (100, 3)
        if rotate_to_dim > 3:
            self.X = self.rotate_to_dim(rotate_to_dim)
        self.labels = np.repeat(
            np.eye(n_distributions), n_points_per_distribution, axis=0
        )
        self.labels = np.argmax(self.labels,-1)

    def _unroll_t(self, t):
        t = t.flatten()  # (100,)
        return 0.5 * ((np.sqrt(t**2 + 1) * t) + np.arcsinh(t)).reshape(
            1, -1
        )  # (1, 100)

    def get_geodesic(self):
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
            self.graph = graphtools.Graph(self.X, use_pygsp=True)
        return self.graph
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SwissRoll(n_distributions=100, n_points_per_distribution=50, width=10.0, noise=0.05, manifold_noise=0.05, random_state=42, rotate_to_dim=5)
    data = dataset.X
    labels = dataset.labels
    gt_distance = dataset.get_geodesic()
    print("Data shape:", dataset.X.shape)
    print("Labels shape:", dataset.labels.shape)
    # 3d plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,1], data[:,2], data[:,3], c=labels, cmap='tab20')
    plt.savefig("swiss_roll.png", bbox_inches='tight')
    
