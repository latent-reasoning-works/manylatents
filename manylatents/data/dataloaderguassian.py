from typing import Union, List, Optional, Tuple
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from gaussian_blobs import GaussianBlobDataset  # Assure-toi que ce fichier est bien accessible


class GaussianBlobs(LightningDataModule):
    def __init__(self,
                 batch_size: int = 128,
                 test_split: float = 0.2,
                 num_workers: int = 0,
                 n_samples: Union[int, List[int]] = 100,
                 n_features: int = 2,
                 centers: Optional[Union[int, np.ndarray, List[List[float]]]] = None,
                 cluster_std: Union[float, List[float]] = 1.0,
                 center_box: Tuple[float, float] = (-10.0, 10.0),
                 shuffle: bool = True,
                 random_state: Optional[int] = 42,  # none but or 42 
                 return_centers: bool = False,
                 mode: str = 'full'):
        super().__init__()
        
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers

        #specific to gaussian blobs
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
        pass

    def setup(self, stage: str = None):
        if self.mode == "full":
            self.train_dataset = GaussianBlobDataset(
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

        elif self.mode == "split":
            self.dataset = GaussianBlobDataset(
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
    dm = GaussianBlobs(
        batch_size=64,
        test_split=0.2,
        num_workers=0,
        n_samples=[100, 150, 200],
        n_features=2,
        centers=[[0, 0], [5, 5], [-5, -5]],
        cluster_std=[0.5, 1.0, 1.5],
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=42,
        return_centers=True,
        mode='split',
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch['data'].shape)
    print("  y shape:", train_batch['metadata'].shape)

    print("Validation batch:")
    print("  x shape:", val_batch['data'].shape)
    print("  y shape:", val_batch['metadata'].shape)
