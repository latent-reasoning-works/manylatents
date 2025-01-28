from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .hgdp_dataset import HGDPDataset

class HGDPDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Human Genome Diversity Project (HGDP) dataset.
    """

    def __init__(
        self, 
        plink_prefix: str, 
        metadata_path: str, 
        batch_size: int = 32, 
        num_workers: int = 4, 
        mode: str = 'genotypes',
        mmap_mode: Optional[str] = None
    ):
        """
        Initializes the HGDPModule with configuration parameters.

        Args:
            plink_prefix (str): Path to the PLINK file prefix (excluding extensions).
            metadata_path (str): Path to the metadata CSV file.
            batch_size (int): The number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            mode (str): Determines the type of data returned ('genotypes' or 'pca').
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
        """
        super().__init__()
        self.plink_prefix = plink_prefix
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.mmap_mode = mmap_mode

        self.dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): One of 'fit', 'validate', 'test', or 'predict'. Default is None.
        """
        self.dataset = HGDPDataset(
            plink_prefix=self.plink_prefix,
            metadata_path=self.metadata_path,
            mode=self.mode,
            mmap_mode=self.mmap_mode
        )

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training."""
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation."""
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for testing."""
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
