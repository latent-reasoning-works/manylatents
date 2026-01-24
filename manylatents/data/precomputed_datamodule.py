import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .precomputed_dataset import PrecomputedDataset, InMemoryDataset, MultiChannelDataset

class PrecomputedDataModule(LightningDataModule):
    """
    DataModule for loading precomputed embeddings from files or directories,
    or from in-memory numpy arrays.

    Supports:
    - Single files and multiple files from SaveEmbeddings
    - Multi-channel embeddings via `channels` parameter (HDF5, .pt, or directory)

    Multi-channel mode:
        When `channels` is provided, loads multiple embedding channels and
        provides `get_embeddings()` method returning Dict[str, Tensor].

    Example (single-channel, backward compatible):
        >>> dm = PrecomputedDataModule(path="embeddings.csv")

    Example (multi-channel):
        >>> dm = PrecomputedDataModule(
        ...     path="embeddings/",
        ...     channels=["dna_evo2", "protein_esm3"],
        ... )
        >>> dm.setup()
        >>> embs = dm.get_embeddings()  # {"dna_evo2": Tensor, "protein_esm3": Tensor}
    """
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        channels: Optional[List[str]] = None,
        batch_size: int = 128,
        num_workers: int = 0,
        label_col: str = None,
        labels_path: Optional[str] = None,
        mode: str = 'full',
        test_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        # Ignore 'data' to prevent Lightning from trying to save the whole array in checkpoints
        self.save_hyperparameters(ignore=['data'])

        if path is None and data is None:
            raise ValueError("PrecomputedDataModule requires either a 'path' or 'data' argument.")
        if path is not None and data is not None:
            raise ValueError("You can only provide 'path' or 'data', not both.")

        # Store the data tensor if provided
        if data is not None:
            if isinstance(data, np.ndarray):
                self.data_tensor = torch.from_numpy(data).float()
            elif isinstance(data, torch.Tensor):
                self.data_tensor = data.float()
            else:
                raise TypeError(f"Unsupported data type: {type(data)}. Expected np.ndarray or torch.Tensor.")
        else:
            self.data_tensor = None

        self.channels = channels
        self._channel_embeddings: Dict[str, torch.Tensor] = {}
        self._labels: Optional[np.ndarray] = None
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if self.data_tensor is not None:
            # In-memory data path: use InMemoryDataset for EmbeddingOutputs compatibility
            full_dataset = InMemoryDataset(self.data_tensor)
        elif self.channels is not None:
            # Multi-channel mode: load each channel from path
            full_dataset = self._setup_multi_channel()
        else:
            # File-based path: use PrecomputedDataset
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

    def _setup_multi_channel(self):
        """Load multi-channel embeddings from directory or HDF5."""
        path = Path(self.hparams.path)

        for channel in self.channels:
            # Try different file patterns
            channel_path = None
            for pattern in [f"{channel}.pt", f"{channel}.npy", f"{channel}.csv", channel]:
                candidate = path / pattern
                if candidate.exists():
                    channel_path = candidate
                    break

            if channel_path is None:
                raise FileNotFoundError(f"Channel '{channel}' not found in {path}")

            # Load based on file type
            if channel_path.suffix == ".pt":
                data = torch.load(channel_path)
                if isinstance(data, dict):
                    emb = data.get("embeddings", data.get("data"))
                    if emb is None:
                        emb = next(iter(data.values()))  # Take first tensor
                else:
                    emb = data
            elif channel_path.suffix == ".npy":
                emb = torch.from_numpy(np.load(channel_path)).float()
            elif channel_path.suffix == ".csv":
                import pandas as pd
                emb = torch.tensor(pd.read_csv(channel_path).values, dtype=torch.float32)
            elif channel_path.is_dir():
                # Load from directory
                ds = PrecomputedDataset(str(channel_path))
                emb = ds.data
            else:
                raise ValueError(f"Unsupported file type: {channel_path}")

            self._channel_embeddings[channel] = emb

        # Load labels if path provided
        if self.hparams.labels_path:
            labels_path = Path(self.hparams.labels_path)
            if labels_path.suffix == ".npy":
                self._labels = np.load(labels_path)
            elif labels_path.suffix == ".pt":
                data = torch.load(labels_path)
                self._labels = data.get("labels", data).numpy() if isinstance(data, dict) else data.numpy()

        # Create dataset with concatenated embeddings for compatibility
        return MultiChannelDataset(self._channel_embeddings, self._labels)

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

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Return dict of channel_name -> embeddings tensor.

        Only available in multi-channel mode (when `channels` was provided).

        Returns:
            Dict mapping channel names to embedding tensors.
            All tensors have same first dimension (aligned samples).

        Raises:
            ValueError: If not in multi-channel mode.
        """
        if not self._channel_embeddings:
            if self.channels is not None:
                self.setup()
            else:
                raise ValueError(
                    "get_embeddings() only available in multi-channel mode. "
                    "Provide `channels` parameter when creating PrecomputedDataModule."
                )
        return self._channel_embeddings

    def get_labels(self) -> Optional[np.ndarray]:
        """Return labels array if available."""
        if self._labels is not None:
            return self._labels
        # Try to get from dataset
        if hasattr(self.train_dataset, 'get_labels'):
            return self.train_dataset.get_labels()
        return None

    def get_tensor(self) -> torch.Tensor:
        """Return concatenated embeddings tensor for LatentModule compatibility.

        In multi-channel mode, concatenates all channels.
        In single-channel mode, returns the embeddings tensor.
        """
        if self._channel_embeddings:
            # Multi-channel: concatenate in channel order
            tensors = [self._channel_embeddings[ch] for ch in self.channels]
            return torch.cat(tensors, dim=-1)
        elif self.data_tensor is not None:
            return self.data_tensor
        elif self.train_dataset is not None:
            return self.train_dataset.data
        else:
            self.setup()
            return self.train_dataset.data