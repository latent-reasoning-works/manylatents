import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .synthetic_dataset import DLAtree
from typing import Union, List, Optional, Dict


class DLATreeDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the DLATree Dataset. 
    """
    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        n_branch: int = 20,
        branch_lengths: Union[List[int], int, None] = None, 
        rand_multiplier: float = 2.0,
        gap_multiplier: float = 0.0,
        sigma: float = 4.0,
        random_state: int = 42,
        n_dim: int = 3,
        disconnect_branches: Optional[List[int]] = [5,15],
        sampling_density_factors: Optional[Dict[int, float]] = None,
        precomputed_path: str = None,
        mmap_mode: str = None,
        mode: str = 'full',
    ):
        """
        Initialize the DLATreeDataModule with configuration parameters for data loading
        and synthetic data generation.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        n_dim : int, default=100
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

        mode : str, default='full'
            Mode for dataset train/test seperation. 
            If 'full', the entire dataset is used as both training and test set (unsplit).
            If 'split', the dataset is randomly split into training and test subsets based on `test_split`.
        """
        super().__init__()
        
            
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers

        # DLAtree-specific
        self.n_dim = n_dim
        self.n_branch = n_branch
        self.branch_lengths = branch_lengths
        self.rand_multiplier = rand_multiplier
        self.gap_multiplier = gap_multiplier
        self.random_state = random_state
        self.sigma = sigma
        self.disconnect_branches = disconnect_branches
        self.sampling_density_factors = sampling_density_factors

        self.mode = mode
        self.precomputed_path = precomputed_path
        self.mmap_mode = mmap_mode

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = DLAtree(
                n_dim=self.n_dim,
                n_branch=self.n_branch,
                branch_lengths=self.branch_lengths,
                rand_multiplier=self.rand_multiplier,
                gap_multiplier=self.gap_multiplier,
                random_state=self.random_state,
                sigma=self.sigma,
                disconnect_branches=self.disconnect_branches,
                sampling_density_factors=self.sampling_density_factors,
                precomputed_path=self.precomputed_path,
                mmap_mode=self.mmap_mode,
            )
            self.test_dataset = self.train_dataset

        elif self.mode == 'split':
            self.dataset = DLAtree(
                n_dim=self.n_dim,
                n_branch=self.n_branch,
                branch_lengths=self.branch_lengths,
                rand_multiplier=self.rand_multiplier,
                gap_multiplier=self.gap_multiplier,
                random_state=self.random_state,
                sigma=self.sigma,
                disconnect_branches=self.disconnect_branches,
                sampling_density_factors=self.sampling_density_factors,
                precomputed_path=self.precomputed_path,
                mmap_mode=self.mmap_mode,
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

    from synthetic_dataset import DLAtree

    # Initialize DataModule
    tree = DLATreeDataModule(
        batch_size=64,
        test_split=0.2,
        n_dim=100,
        n_branch=8,
        branch_lengths=100,
        rand_multiplier=2.0,
        gap_multiplier=10.0,
        sigma=0.5,
        disconnect_branches=[3],
        random_state=0,
        mode='split', 
        precomputed_path=None,
        mmap_mode=None,
    )

    # Setup datasets
    tree.setup()

    # Load one batch from train and val
    train_loader = tree.train_dataloader()
    val_loader = tree.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch['data'].shape)
    print("  y shape:", train_batch['metadata'].shape)

    print("Validation batch:")
    print("  x shape:", val_batch['data'].shape)
    print("  y shape:", val_batch['metadata'].shape)