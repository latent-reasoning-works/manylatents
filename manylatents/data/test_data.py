import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    """
    A simple dataset for testing purposes.
    """
    def __init__(self, n_samples=10, n_features=5):
        self.data = torch.tensor(np.random.rand(n_samples, n_features), dtype=torch.float32)
        self.labels = torch.tensor(np.zeros(n_samples), dtype=torch.float32)  # Dummy labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx], 
            "label": self.labels[idx]
        }

class TestDataModule(LightningDataModule):
    """
    A DataModule for the TestDataset.
    """
    def __init__(self, n_samples=10, n_features=5, batch_size=4, num_workers=0, shuffle_traindata=True):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.mode = "split"  # Add mode attribute for compatibility

    def setup(self, stage=None):
        self.dataset = TestDataset(self.n_samples, self.n_features)
        self.test_dataset = self.dataset  # Add test_dataset attribute for compatibility

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle_traindata, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)