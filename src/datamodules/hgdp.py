
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .hgdp_dataset import HGDPDataset


class HGDPDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Human Genome Diversity Project (HGDP) dataset.
    """

    def __init__(
        self,
        files: dict,
        mode: str,
        batch_size: int,
        num_workers: int,
        cache_dir: str,
        mmap_mode: str = None

    ):
        """
        Initializes the HGDPModule with configuration parameters.

        Args:
            filenames (dict): Dictionary containing paths for plink and metadata files.
            batch_size (int): The number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            mode (str): Determines the type of data returned ('genotypes' or 'pca').
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
        """
        super().__init__()
        self.files = files
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.mmap_mode = mmap_mode
        self.dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.
        """
        self.dataset = HGDPDataset(
            files=self.files,
            cache_dir=self.cache_dir,
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
