import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .synthetic_dataset import Torus

class TorusDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Torus Dataset. 
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        n_points: int = 5000,
        noise: float = 0.1,
        major_radius: float = 3.0,
        minor_radius: float = 1.0,
        random_state: int = 42,
        rotate_to_dim: int = 3,
        mode: str = 'split',
    ):
        """
        Initialize the TorusDataModule with configuration parameters for data loading
        and synthetic data generation.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        n_points : int, default=5000
            Total number of points to generate on the torus surface.

        noise : float, default=0.1
            Standard deviation of isotropic Gaussian noise added to each data point.

        major_radius : float, default=3.0
            Major radius of the torus (distance from center to tube center).

        minor_radius : float, default=1.0
            Minor radius of the torus (radius of the tube).

        random_state : int, default=42
            Seed for the random number generator to ensure reproducibility.

        rotate_to_dim : int, default=3
            Target dimension for rotation. Rotation is only applied if this value is greater than 3.

        mode : str, default='split'
            Mode for dataset train/test separation. 
            If 'full', the entire dataset is used as both training and test set (unsplit).
            If 'split', the dataset is randomly split into training and test subsets based on `test_split`.
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers

        # Torus specific
        self.n_points = n_points
        self.noise = noise
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.random_state = random_state
        self.rotate_to_dim = rotate_to_dim

        self.mode = mode

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = Torus(
                n_points=self.n_points,
                noise=self.noise,
                major_radius=self.major_radius,
                minor_radius=self.minor_radius,
                random_state=self.random_state,
                rotate_to_dim=self.rotate_to_dim,
            )
            self.test_dataset = self.train_dataset

        elif self.mode == 'split':
            self.dataset = Torus(
                n_points=self.n_points,
                noise=self.noise,
                major_radius=self.major_radius,
                minor_radius=self.minor_radius,
                random_state=self.random_state,
                rotate_to_dim=self.rotate_to_dim,
            )

            test_size = int(len(self.dataset) * self.test_split)
            train_size = len(self.dataset) - test_size

            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    from synthetic_dataset import Torus
    # Initialize DataModule
    torus = TorusDataModule(
        batch_size=64,
        test_split=0.2,
        n_points=1000,
        noise=0.05,
        major_radius=3.0,
        minor_radius=1.0,
        random_state=123,
        rotate_to_dim=5,
    )

    # Setup datasets
    torus.setup()

    # Load one batch from train and val
    train_loader = torus.train_dataloader()
    val_loader = torus.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch['data'].shape)
    print("  y shape:", train_batch['metadata'].shape)

    print("Validation batch:")
    print("  x shape:", val_batch['data'].shape)
    print("  y shape:", val_batch['metadata'].shape)
