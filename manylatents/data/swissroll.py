import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from typing import Union, List
from .synthetic_dataset import SwissRoll, SwissRollGap

class SwissRollDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the SwissRoll Dataset. 
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        shuffle_traindata: bool = True,
        n_distributions: int = 100,
        n_points_per_distribution: Union[int, List[int]] = 50,
        noise: float = 0.1,
        manifold_noise: float = 0.1,
        width: float = 10.0,
        random_state: int = 42,
        rotate_to_dim: int = 3,
        mode: str = 'full',
        use_gap: bool = False,
        n_gaps: int = 3,
        gap_fraction: float = 0.12,
    ):
        """
        Initialize the SwissRollDataModule with configuration parameters for data loading
        and synthetic data generation.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        n_distributions : int, default=100
            Number of independent Gaussian distributions along the Swiss roll manifold.

        n_points_per_distribution : int or list, default=50
            If int: number of samples drawn from each Gaussian distribution (equal-sized blobs).
            If list/sequence: per-distribution counts; length must equal `n_distributions`.

        noise : float, default=0.1
            Global Gaussian noise added to all data points for variability.

        manifold_noise : float, default=0.1
            Local noise controlling the spread of points within each Gaussian distribution.

        width : float, default=10.0
            Width of the Swiss roll, determining the spread of the manifold in the vertical axis.

        random_state : int, default=42
            Seed for the random number generator to ensure reproducibility.

        rotate_to_dim : int, default=3
            Target dimension for rotation. Rotation is only applied if this value is greater than 3.
            The default of 3 keeps the Swiss roll in 3D space.

        mode : str, default='full'
            Mode for dataset train/test seperation. 
            If 'full', the entire dataset is used as both training and test set (unsplit).
            If 'split', the dataset is randomly split into training and test subsets based on `test_split`.
        """
        super().__init__()
        
            
        self.batch_size = batch_size
        self.shuffle_traindata = shuffle_traindata
        self.test_split = test_split
        self.num_workers = num_workers

        # SwissRoll-specific
        self.n_distributions = n_distributions
        self.n_points_per_distribution = n_points_per_distribution
        self.noise = noise
        self.manifold_noise = manifold_noise
        self.width = width
        self.random_state = random_state
        self.rotate_to_dim = rotate_to_dim

        self.mode = mode
        # If True, use the SwissRollGap variation (manifold with large gaps)
        self.use_gap = use_gap
        # Parameters controlling the gaps when use_gap=True
        self.n_gaps = n_gaps
        self.gap_fraction = gap_fraction

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            if self.use_gap:
                # SwissRollGap accepts per-distribution counts (list) or an int
                self.train_dataset = SwissRollGap(
                    n_distributions=self.n_distributions,
                    n_points_per_distribution=self.n_points_per_distribution,
                    noise=self.noise,
                    manifold_noise=self.manifold_noise,
                    width=self.width,
                    random_state=self.random_state,
                    rotate_to_dim=self.rotate_to_dim,
                    n_gaps=self.n_gaps,
                    gap_fraction=self.gap_fraction,
                )
            else:
                # SwissRoll accepts either an int or a per-distribution list/array.
                self.train_dataset = SwissRoll(
                    n_distributions=self.n_distributions,
                    n_points_per_distribution=self.n_points_per_distribution,
                    noise=self.noise,
                    manifold_noise=self.manifold_noise,
                    width=self.width,
                    random_state=self.random_state,
                    rotate_to_dim=self.rotate_to_dim,
                )
            self.test_dataset = self.train_dataset

        elif self.mode == 'split':
            if self.use_gap:
                self.dataset = SwissRollGap(
                    n_distributions=self.n_distributions,
                    n_points_per_distribution=self.n_points_per_distribution,
                    noise=self.noise,
                    manifold_noise=self.manifold_noise,
                    width=self.width,
                    random_state=self.random_state,
                    rotate_to_dim=self.rotate_to_dim,
                    n_gaps=self.n_gaps,
                    gap_fraction=self.gap_fraction,
                )
            else:
                # SwissRoll accepts either an int or a per-distribution list/array.
                self.dataset = SwissRoll(
                    n_distributions=self.n_distributions,
                    n_points_per_distribution=self.n_points_per_distribution,
                    noise=self.noise,
                    manifold_noise=self.manifold_noise,
                    width=self.width,
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
            shuffle=self.shuffle_traindata,
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
    from .synthetic_dataset import SwissRoll, SwissRollGap
    # for testing only: python -m manylatents.data.swissroll

    print("Testing SwissRollDataModule with SwissRoll (no gaps):")
    sr = SwissRollDataModule(
        batch_size=64,
        test_split=0.2,
        n_distributions=10,
        n_points_per_distribution=30,
        noise=0.05,
        manifold_noise=0.05,
        width=5.0,
        random_state=123,
        rotate_to_dim=5,
        mode='split',
        use_gap=False,
    )
    sr.setup()
    train_loader = sr.train_dataloader()
    val_loader = sr.val_dataloader()
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print("SwissRoll train batch x shape:", train_batch['data'].shape)
    print("SwissRoll train batch y shape:", train_batch['metadata'].shape)
    print("SwissRoll val batch x shape:", val_batch['data'].shape)
    print("SwissRoll val batch y shape:", val_batch['metadata'].shape)

    print("\nTesting SwissRollDataModule with SwissRollGap (gaps enabled):")
    sr_gap = SwissRollDataModule(
        batch_size=64,
        test_split=0.2,
        n_distributions=10,
        n_points_per_distribution=30,
        noise=0.05,
        manifold_noise=0.05,
        width=5.0,
        random_state=123,
        rotate_to_dim=5,
        precomputed_path=None,
        mmap_mode=None,
        mode='split',
        use_gap=True,
        n_gaps=2,
        gap_fraction=0.2,
    )
    sr_gap.setup()
    train_loader_gap = sr_gap.train_dataloader()
    val_loader_gap = sr_gap.val_dataloader()
    train_batch_gap = next(iter(train_loader_gap))
    val_batch_gap = next(iter(val_loader_gap))
    print("SwissRollGap train batch x shape:", train_batch_gap['data'].shape)
    print("SwissRollGap train batch y shape:", train_batch_gap['metadata'].shape)
    print("SwissRollGap val batch x shape:", val_batch_gap['data'].shape)
    print("SwissRollGap val batch y shape:", val_batch_gap['metadata'].shape)

    # Print summary of gap effect
    print("\nSwissRoll total points:", len(sr.train_dataset))
    print("SwissRollGap total points:", len(sr_gap.train_dataset))