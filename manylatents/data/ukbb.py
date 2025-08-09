# ukbb_data_module.py
from typing import Optional, Dict, Any, Union

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .ukbb_dataset import UKBBDataset


class UKBBDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for UK Biobank (UKBB).

    The API is now aligned with HGDPDataModule:
      • same constructor signature (incl. optional `metadata`)
      • identical `mode` logic ("full" or "split")
      • collate_fn returns a dict → {"data": …, "metadata": …, "precomputed": …}
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        cache_dir: str,
        files: Optional[Dict[str, str]] = None,
        mmap_mode: Optional[str] = None,
        precomputed_path: Optional[str] = None,
        metadata: Optional[pd.DataFrame] = None,
        delimiter: str = ",",
        filter_qc: bool = False,
        filter_related: bool = False,
        balance_filter: Union[bool, float] = False,
        include_do_not_know: bool = False,
        test_all: bool = False,
        remove_recent_migration: bool = False,
        mode: Optional[str] = None,
    ):
        super().__init__()

        # — store init kwargs —
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.files = files or {}
        self.mmap_mode = mmap_mode
        self.precomputed_path = precomputed_path
        self.metadata = metadata          # may stay None
        self.delimiter = delimiter
        self.filter_qc = filter_qc
        self.filter_related = filter_related
        self.balance_filter = balance_filter
        self.include_do_not_know = include_do_not_know
        self.test_all = test_all
        self.remove_recent_migration = remove_recent_migration
        self.mode = mode or "split"       # default split if caller forgets

    # --------------------------------------------------------------------- #
    # Lightning hooks
    # --------------------------------------------------------------------- #
    def prepare_data(self) -> None:
        """No downloading required – datasets take care of conversion/caching."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Instantiate UKBBDataset(s) depending on `self.mode`.

        • mode == "full": one dataset used for both train & test.
        • mode == "split": independent train/test splits.
        """
        common_kwargs: Dict[str, Any] = dict(
            files=self.files,
            cache_dir=self.cache_dir,
            mmap_mode=self.mmap_mode,
            precomputed_path=self.precomputed_path,
            metadata=self.metadata,
            delimiter=self.delimiter,
            filter_qc=self.filter_qc,
            filter_related=self.filter_related,
            balance_filter=self.balance_filter,
            include_do_not_know=self.include_do_not_know,
            test_all=self.test_all,
            remove_recent_migration=self.remove_recent_migration,
        )

        if self.mode == "full":
            self.train_dataset = UKBBDataset(data_split="full", **common_kwargs)
            self.test_dataset = self.train_dataset   # same object
        elif self.mode == "split":
            self.train_dataset = UKBBDataset(data_split="train", **common_kwargs)
            self.test_dataset = UKBBDataset(data_split="test", **common_kwargs)
        else:
            raise ValueError("`mode` must be 'full' or 'split'.")

    # --------------------------------------------------------------------- #
    # Dataloaders
    # --------------------------------------------------------------------- #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        # For now we validate on the training split (same as HGDPDataModule).
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _collate_fn(batch):
        """
        Make the batch format identical to HGDPDataModule.

        * Accepts either {'data': …} or legacy {'raw': …}
        * Adds 'precomputed' key only when present / not-None
        """
        sample0 = batch[0]

        # decide which key holds the primary array
        data_key = "data" if "data" in sample0 else "raw"
        data = torch.stack(
            [torch.as_tensor(s[data_key], dtype=torch.float32) for s in batch]
        )

        meta = [s["metadata"] for s in batch]

        out = {"data": data, "metadata": meta}

        if "precomputed" in sample0 and sample0["precomputed"] is not None:
            pre = torch.stack(
                [torch.as_tensor(s["precomputed"], dtype=torch.float32) for s in batch]
            )
            out["precomputed"] = pre

        return out

    # convenient helper for model dimension introspection
    @property
    def dims(self):
        sample = self.train_dataset[0]
        key = "data" if "data" in sample else "raw"
        return sample[key].shape