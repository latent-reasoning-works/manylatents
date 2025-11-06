import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .anndata_dataset import AnnDataset

class AnnDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule that handles Anndata (.h5ad) files.
    This module is useful when dealing with single-cell RNA-seq datasets stored in Anndata format.
    """

    def __init__(
        self,
        batch_size: int = 128,
        test_split: float = 0.2,
        num_workers: int = 0,
        random_state: int = 42,
        shuffle_traindata: bool = False,
        adata_path: str = None,
        label_key: str = None,
        mode: str = 'full',
    ):
        """
        Initialize the AnnDataModule with configuration parameters for data loading
        and synthetic data generation.

        Parameters
        ----------
        batch_size : int, default=128
            Number of samples per batch used in training and validation data loaders.

        test_split : float, default=0.2
            Fraction of the dataset to allocate to the test set.

        num_workers : int, default=0
            Number of subprocesses to use for data loading in PyTorch's DataLoader.

        random_state : int, default=42
            Random seed for reproducibility when splitting datasets.
        
        shuffle_traindata : bool, default=True
            Whether to shuffle the training data in the DataLoader.

        adata_path : str
            Path to the raw .h5ad file containing the full dataset.

        label_key : str, default="cell_type"
            Key in adata.obs used to retrieve cell type or condition labels.

        mode : str, default='full'
            Mode for dataset train/test seperation. 
            If 'full', the entire dataset is used as both training and test set (unsplit).
            If 'split', the dataset is randomly split into training and test subsets based on `test_split`.
        """
        super().__init__()
        
            
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers
        self.random_state = random_state
        self.shuffle_traindata = shuffle_traindata

        # AnnDataset specific parameters
        self.adata_path = adata_path
        self.label_key = label_key

        self.mode = mode

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        """Prepare data for use (e.g., downloading, saving to disk)."""
        pass

    def setup(self):
        if self.mode == "full":
            self.train_dataset = AnnDataset(adata_path=self.adata_path,
                                              label_key=self.label_key)
            self.test_dataset = self.train_dataset

        elif self.mode == 'split':
            self.dataset = AnnDataset(adata_path=self.adata_path,
                                        label_key=self.label_key)
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
            shuffle=self.shuffle_traindata,
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

    from .anndata_dataset import AnnDataset

    # Initialize DataModule
    EB = AnnDataModule(
        adata_path="./data/scRNAseq/EBT_counts.h5ad",
        label_key="sample_labels",
        batch_size=32,
        test_split=0.2,
        num_workers=4,
        random_state=42,
        mode='split'
    )

    # Setup datasets
    EB.setup()

    # Load one batch from train and val
    train_loader = EB.train_dataloader()
    val_loader = EB.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Train batch:")
    print("  x shape:", train_batch['data'].shape)
    print("  y shape:", len(train_batch['metadata']))

    print("Validation batch:")
    print("  x shape:", val_batch['data'].shape)
    print("  y shape:", len(val_batch['metadata']))