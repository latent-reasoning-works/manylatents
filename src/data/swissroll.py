import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .synthetic_dataset import SwissRoll

class SwissRollDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the SwissRoll Dataset. 
    """

    def __init__(
        self,
        batch_size: int = 128,
        val_split: float = 0.2,
        num_workers: int = 0,
        n_distributions: int = 100,
        n_points_per_distribution: int = 50,
        noise: float = 0.05,
        manifold_noise: float = 0.05,
        width: float = 10.0,
        random_state: int = 42,
        rotate_to_dim: int = 3,
    ):
        """
        Initializes the SwissRollDataModule with configuration parameters.

        Args:
            batch_size (int): Number of samples per batch.
            val_split (float): Fraction of the dataset used for validation.
            num_workers (int): Number of subprocesses to use for data loading.
            n_distributions (int): Number of distributions in the synthetic Swiss Roll.
            n_points_per_distribution (int): Number of samples per distribution.
            noise (float): Amount of additive noise to the final data.
            manifold_noise (float): Noise along the Swiss roll manifold.
            width (float): Width of the Swiss roll (controls spread in Y-axis).
            random_state (int): Random seed for reproducibility.
            rotate_to_dim (int): Output dimension to which the 3D Swiss roll is rotated.
        """
        super().__init__()
        
            
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        # SwissRoll-specific
        self.n_distributions = n_distributions
        self.n_points_per_distribution = n_points_per_distribution
        self.noise = noise
        self.manifold_noise = manifold_noise
        self.width = width
        self.random_state = random_state
        self.rotate_to_dim = rotate_to_dim

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: str = None):
        self.dataset = SwissRoll(
            n_distributions=self.n_distributions,
            n_points_per_distribution=self.n_points_per_distribution,
            noise=self.noise,
            manifold_noise=self.manifold_noise,
            width=self.width,
            random_state=self.random_state,
            rotate_to_dim=self.rotate_to_dim,
        )

        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
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
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
if __name__ == "__main__":

    # Initialize DataModule
    sr = SwissRollDataModule(
        batch_size=64,
        val_split=0.2,
        n_distributions=10,
        n_points_per_distribution=30,
        noise=0.05,
        manifold_noise=0.05,
        width=5.0,
        random_state=123,
        rotate_to_dim=5,
    )

    # Setup datasets
    sr.setup()

    # Load one batch from train and val
    train_loader = sr.train_dataloader()
    val_loader = sr.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch[0].shape)
    print("  y shape:", train_batch[1].shape)

    print("Validation batch:")
    print("  x shape:", val_batch[0].shape)
    print("  y shape:", val_batch[1].shape)