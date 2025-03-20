
import torch
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
        batch_size: int,
        num_workers: int,
        cache_dir: str,
        mmap_mode: str = None,
        precomputed_path: str = None,
        delimiter: str = ",",
        filter_related: bool = True,
        mode: str = None,

    ):
        """
        Initializes the HGDPModule with configuration parameters.
        
        Args:
            files (dict): Paths for PLINK and metadata files.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
            cache_dir (str): Directory for caching data.
            mmap_mode (Optional[str]): Memory-mapping mode.
            precomputed_path (Optional[str]): Path to precomputed embeddings.
            delimiter (Optional[str]): Delimiter for CSV files.
            filter_related (bool): Whether to filter related samples.
            mode (str): 'split' or 'full' mode. Former splits data into train/test according to indices, 
            latter uses all data for both fit and transform operations.


        """
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.mmap_mode = mmap_mode
        self.delimiter = delimiter
        self.precomputed_path = precomputed_path
        self.filter_related = filter_related
        self.mode = mode

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.
        """
        if self.mode == "full":
            self.train_dataset = HGDPDataset(
                files=self.files,
                cache_dir=self.cache_dir,
                mmap_mode=self.mmap_mode,
                precomputed_path=self.precomputed_path,
                delimiter=self.delimiter,
                filter_related=self.filter_related,
                data_split='full',
            )

            self.test_dataset = self.train_dataset
            
        elif self.mode == 'split':
            self.train_dataset = HGDPDataset(
                files=self.files,
                cache_dir=self.cache_dir,
                mmap_mode=self.mmap_mode,
                precomputed_path=self.precomputed_path,
                delimiter=self.delimiter,
                filter_related=self.filter_related,
                data_split='train',
            )

            self.test_dataset = HGDPDataset(
                files=self.files,
                cache_dir=self.cache_dir,
                mmap_mode=self.mmap_mode,
                precomputed_path=self.precomputed_path,
                delimiter=self.delimiter,
                filter_related=self.filter_related,
                data_split='test',
            )

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Use 'full' or 'split'.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        # calls train directly, manage splitting logic later if required
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=self._collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=self._collate_fn)

    @staticmethod
    def _collate_fn(batch):
        raw_samples = [torch.tensor(sample["raw"], dtype=torch.float32) for sample in batch]
        precomputed_samples = None
        if batch[0]["precomputed"] is not None:
            precomputed_samples = [torch.tensor(sample["precomputed"], dtype=torch.float32) for sample in batch]
            precomputed_samples = torch.stack(precomputed_samples)
        metadata = [sample["metadata"] for sample in batch]
        return torch.stack(raw_samples), precomputed_samples, metadata
    
    @property
    def dims(self):
        sample = self.train_dataset[0]
        return sample["raw"].shape
