import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx]
        }
    
    def get_labels(self):
        """Return the labels for the dataset."""
        return self.labels


class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, num_samples: int = 100, input_dim: int = 10):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_dim = input_dim
        
    def setup(self, stage=None):
        # Create dummy data
        data = torch.randn(self.num_samples, self.input_dim)
        labels = torch.randint(0, 3, (self.num_samples,))
        
        # Split into train/test
        train_size = int(0.8 * self.num_samples)
        self.train_dataset = DummyDataset(data[:train_size], labels[:train_size])
        self.test_dataset = DummyDataset(data[train_size:], labels[train_size:])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False) 