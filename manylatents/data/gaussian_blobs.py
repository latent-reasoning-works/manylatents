
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .synthetic_dataset import GaussianBlobs

logger = logging.getLogger(__name__)

class GaussianBlobDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Gaussian Blob Dataset.
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        n_samples: Union[int, List[int]] = 100,
        n_features: int = 2,
        centers: Optional[Union[int, np.ndarray, List[List[float]]]] = None,
        cluster_std: Union[float, List[float]] = 1.0,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        shuffle: bool = True,
        shuffle_traindata: bool = False,    
        random_state: Optional[int] = 42,
        return_centers: bool = False,
        mode: str = 'split',
    ):
        """
        Initialize the GaussianBlobDataModule with configuration parameters for data loading
        and synthetic data generation.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        n_samples : int or list of int, default=100
            Number of samples to generate, or list of samples per center.

        n_features : int, default=2
            Number of features for each sample.

        centers : int, array-like or None, default=None
            Number of centers to generate, or fixed center locations.

        cluster_std : float or list of float, default=1.0
            Standard deviation of the clusters.

        center_box : tuple of float, default=(-10.0, 10.0)
            Bounding box for each cluster center when centers are generated at random.

        shuffle : bool, default=True
            Shuffle the samples.

        random_state : int, default=42
            Random state for reproducibility.

        return_centers : bool, default=False
            Whether to return the centers.

        mode : str, default='split'
            Mode for dataset train/test separation.
            If 'full', the entire dataset is used as both training and test set (unsplit).
            If 'split', the dataset is randomly split into training and test subsets based on `test_split`.
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers

        # Gaussian Blob specific
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.shuffle = shuffle
        self.random_state = random_state
        self.return_centers = return_centers

        self.mode = mode

        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = GaussianBlobs(
                n_samples=self.n_samples,
                n_features=self.n_features,
                centers=self.centers,
                cluster_std=self.cluster_std,
                center_box=self.center_box,
                shuffle=self.shuffle,
                random_state=self.random_state,
                return_centers=self.return_centers,
            )
            self.test_dataset = self.train_dataset

        elif self.mode == 'split':
            self.dataset = GaussianBlobs(
                n_samples=self.n_samples,
                n_features=self.n_features,
                centers=self.centers,
                cluster_std=self.cluster_std,
                center_box=self.center_box,
                shuffle=self.shuffle,
                random_state=self.random_state,
                return_centers=self.return_centers,
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
    from synthetic_dataset import GaussianBlobs
    # Initialize DataModule
    gaussian_blobs = GaussianBlobDataModule(
        batch_size=64,
        test_split=0.2,
        n_samples=1000,
        n_features=2,
        centers=5,
        cluster_std=1.0,
        random_state=123,
    )

    # Setup datasets
    gaussian_blobs.setup()

    # Load one batch from train and val
    train_loader = gaussian_blobs.train_dataloader()
    val_loader = gaussian_blobs.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch['data'].shape)
    print("  y shape:", train_batch['metadata'].shape)

    print("Validation batch:")
    print("  x shape:", val_batch['data'].shape)
    print("  y shape:", val_batch['metadata'].shape)