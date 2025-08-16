import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .precomputed_dataset import PrecomputedDataset

class PrecomputedDataModule(LightningDataModule):
    """
    A generic DataModule for loading data from a single pre-computed file.
    """
    def __init__(
        self,
        path: str,
        batch_size: int = 128,
        num_workers: int = 0,
        label_col: str = None,
        mode: str = 'full',
        test_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
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