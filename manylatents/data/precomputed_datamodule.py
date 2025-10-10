import torch
import numpy as np
from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .precomputed_dataset import PrecomputedDataset, InMemoryDataset

class PrecomputedDataModule(LightningDataModule):
    """
    DataModule for loading precomputed embeddings from files or directories,
    or from in-memory numpy arrays.
    Supports both single files and multiple files from SaveEmbeddings.
    """
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        batch_size: int = 128,
        num_workers: int = 0,
        label_col: str = None,
        mode: str = 'full',
        test_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        # Ignore 'data' to prevent Lightning from trying to save the whole array in checkpoints
        self.save_hyperparameters(ignore=['data'])

        if path is None and data is None:
            raise ValueError("PrecomputedDataModule requires either a 'path' or 'data' argument.")
        if path is not None and data is not None:
            raise ValueError("You can only provide 'path' or 'data', not both.")

        # Store the data tensor if provided
        if data is not None:
            self.data_tensor = torch.from_numpy(data).float()
        else:
            self.data_tensor = None

        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if self.data_tensor is not None:
            # In-memory data path: use InMemoryDataset for EmbeddingOutputs compatibility
            full_dataset = InMemoryDataset(self.data_tensor)
        else:
            # File-based path: use PrecomputedDataset
            full_dataset = PrecomputedDataset(path=self.hparams.path, label_col=self.hparams.label_col)

        if self.hparams.mode == 'full':
            self.train_dataset = full_dataset
            self.test_dataset = full_dataset
        elif self.hparams.mode == 'split':
            test_size = int(len(full_dataset) * self.hparams.test_split)
            train_size = len(full_dataset) - test_size
            self.train_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, test_size],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )
        else:
            raise ValueError(f"Mode '{self.hparams.mode}' is not supported. Use 'full' or 'split'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True
        )
        
    def val_dataloader(self) -> DataLoader:
        # Validation uses the training set
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )