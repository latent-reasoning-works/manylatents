from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

# from .hgdp_dataset import HGDPDataset  # Uncomment when dataset logic is implemented

class HGDPModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Human Genome Diversity Project (HGDP) dataset.

    Encapsulates all data-related operations, including downloading, preprocessing, and
    preparing data loaders.
    """

    def __init__(self, batch_size: int = 32, num_workers: int = 4, data_dir: str = "./data"):
        """
        Initializes the HGDPModule with configuration parameters.

        Args:
            batch_size (int): The number of samples per batch. Default is 32.
            num_workers (int): Number of subprocesses to use for data loading. Default is 4.
            data_dir (str): Path to the dataset directory. Default is "./data".
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): One of 'fit', 'validate', 'test', or 'predict'. Default is None.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training."""
        return DataLoader(None, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation."""
        return DataLoader(None, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for testing."""
        return DataLoader(None, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Return DataLoader for prediction."""
        return DataLoader(None, batch_size=self.batch_size, num_workers=self.num_workers)
