import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from .synthetic_dataset import TuningFork


class TuningForkDataModule(LightningDataModule):
    """LightningDataModule for the TuningFork synthetic dataset."""

    def __init__(
        self,
        n_prong: int = 500,
        handle_prong_ratio: float = 0.2,
        dist_between_prongs: float = 0.3,
        prong_length: float = 3.0,
        handle_length: float = 2.0,
        noise: float = 0.02,
        rotate_to_dim: int = 2,
        random_state: int = 42,
        save_viz: bool = False,
        save_dir: str = "outputs",
        batch_size: int = 128,
        num_workers: int = 0,
        shuffle_traindata: bool = True,
        test_split: float = 0.2,
        mode: str = "full",
    ):
        super().__init__()
        self.n_prong = n_prong
        self.handle_prong_ratio = handle_prong_ratio
        self.dist_between_prongs = dist_between_prongs
        self.prong_length = prong_length
        self.handle_length = handle_length
        self.noise = noise
        self.rotate_to_dim = rotate_to_dim
        self.random_state = random_state
        self.save_viz = save_viz
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_traindata = shuffle_traindata
        self.test_split = test_split
        self.mode = mode

        self.train_dataset = None
        self.test_dataset = None

    def _make_dataset(self):
        return TuningFork(
            n_prong=self.n_prong,
            handle_prong_ratio=self.handle_prong_ratio,
            dist_between_prongs=self.dist_between_prongs,
            prong_length=self.prong_length,
            handle_length=self.handle_length,
            noise=self.noise,
            rotate_to_dim=self.rotate_to_dim,
            random_state=self.random_state,
            save_viz=self.save_viz,
            save_dir=self.save_dir,
        )

    def setup(self, stage=None):
        if self.mode == "full":
            ds = self._make_dataset()
            self.train_dataset = ds
            self.test_dataset = ds
        elif self.mode == "split":
            ds = self._make_dataset()
            test_size = int(len(ds) * self.test_split)
            train_size = len(ds) - test_size
            self.train_dataset, self.test_dataset = random_split(
                ds,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_traindata,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
