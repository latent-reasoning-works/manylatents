import json
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class InMemoryDataset(Dataset):
    """
    PyTorch Dataset for in-memory numpy arrays.

    This dataset wraps a numpy array and returns samples in the EmbeddingOutputs format,
    ensuring compatibility with the rest of the manyLatents pipeline.

    EmbeddingOutputs format:
        {
            "embeddings": tensor,  # Main embeddings data
            "data": tensor,        # Compatibility alias for embeddings
            "label": tensor,       # Optional labels
        }
    """
    def __init__(self, data_tensor: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            data_tensor (torch.Tensor): The embeddings tensor
            labels (torch.Tensor, optional): Optional labels tensor
        """
        self.data = data_tensor
        self.embedding_outputs = {"embeddings": data_tensor}

        if labels is not None:
            self.embedding_outputs["label"] = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return sample from EmbeddingOutputs format."""
        sample = {}
        for key, value in self.embedding_outputs.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value[idx]
            else:
                sample[key] = value

        # Maintain compatibility - ensure 'data' key points to main embeddings
        if "data" not in sample and "embeddings" in sample:
            sample["data"] = sample["embeddings"]

        return sample

    def get_labels(self):
        """Returns all labels for compatibility with plotting callbacks."""
        if "label" in self.embedding_outputs:
            labels = self.embedding_outputs["label"]
            return labels.numpy() if isinstance(labels, torch.Tensor) else labels
        return np.zeros(len(self.data))


class PrecomputedDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed embeddings.

    Supports:
    - Single files: .csv or .npy (legacy format)
    - Multiple files: from SaveEmbeddings with save_additional_outputs=True

    Always returns EmbeddingOutputs format in __getitem__.
    """
    def __init__(self, path: str, label_col: str = None):
        """
        Args:
            path (str): Path to data file (.csv/.npy) or directory with multiple files
            label_col (str, optional): Column name for labels in CSV files
        """
        self.path = path
        self.label_col = label_col
        self.embedding_outputs = {}
        self._load_data()

    def _load_data(self):
        """Load data and convert to EmbeddingOutputs format."""
        logger.info(f"Loading precomputed data from: {self.path}")

        if os.path.isfile(self.path):
            self._load_single_file()
        elif os.path.isdir(self.path):
            self._load_multiple_files()
        else:
            raise ValueError(f"Path not found: {self.path}")

        # Ensure we have embeddings
        if "embeddings" not in self.embedding_outputs:
            raise ValueError("No embeddings found in data")

        self.data = self.embedding_outputs["embeddings"]
        logger.info(f"Successfully loaded embeddings with shape: {self.data.shape}")

    def _load_single_file(self):
        """Load from single .csv or .npy file."""
        path = Path(self.path)

        if path.suffix == '.csv':
            df = pd.read_csv(self.path)

            # Extract labels if specified
            labels = None
            if self.label_col and self.label_col in df.columns:
                labels = torch.tensor(pd.to_numeric(df[self.label_col], errors='coerce').fillna(0).values)
                df = df.drop(columns=[self.label_col])

            # Main embeddings
            embeddings = torch.tensor(df.values, dtype=torch.float32)

            self.embedding_outputs = {"embeddings": embeddings}
            if labels is not None:
                self.embedding_outputs["label"] = labels

        elif path.suffix == '.npy':
            embeddings = torch.tensor(np.load(self.path), dtype=torch.float32)
            self.embedding_outputs = {"embeddings": embeddings}

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_multiple_files(self):
        """Load from directory with multiple files from SaveEmbeddings."""
        path = Path(self.path)

        # Find all relevant files
        embedding_files = {}
        for file_path in path.glob("*"):
            if file_path.is_file():
                stem = file_path.stem
                suffix = file_path.suffix

                if suffix == '.npy':
                    # Load numpy arrays
                    data = np.load(file_path)
                    # Extract key from filename (e.g., "embeddings_exp_20250922_cluster_labels" -> "cluster_labels")
                    if '_' in stem:
                        key = stem.split('_')[-1] if not stem.endswith('embeddings') else 'embeddings'
                    else:
                        key = stem
                    embedding_files[key] = torch.tensor(data, dtype=torch.float32)

                elif suffix == '.json':
                    # Load JSON data
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    key = stem.split('_')[-1] if '_' in stem else stem
                    embedding_files[key] = data

                elif suffix == '.csv':
                    # Load CSV data
                    df = pd.read_csv(file_path)
                    key = stem.split('_')[-1] if '_' in stem else stem
                    embedding_files[key] = torch.tensor(df.values, dtype=torch.float32)

        self.embedding_outputs = embedding_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return sample from EmbeddingOutputs format."""
        # Build sample dictionary with indexed data
        sample = {}
        for key, value in self.embedding_outputs.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value[idx]
            else:
                # For non-tensor data (metadata, etc.), include as-is
                sample[key] = value

        # Maintain compatibility - ensure 'data' key points to main embeddings
        if "data" not in sample and "embeddings" in sample:
            sample["data"] = sample["embeddings"]

        return sample

    def get_labels(self):
        """Returns all labels for compatibility with plotting callbacks."""
        if "label" in self.embedding_outputs:
            labels = self.embedding_outputs["label"]
            return labels.numpy() if isinstance(labels, torch.Tensor) else labels
        return np.zeros(len(self.data))


class MultiChannelDataset(Dataset):
    """
    PyTorch Dataset for multi-channel precomputed embeddings.

    Stores multiple embedding channels and provides samples with all channels
    concatenated. Supports the EmbeddingOutputs format.

    Example:
        >>> channels = {"dna": dna_tensor, "protein": protein_tensor}
        >>> ds = MultiChannelDataset(channels, labels=labels_array)
        >>> sample = ds[0]  # {"embeddings": concat, "data": concat, "dna": ..., "protein": ...}
    """
    def __init__(
        self,
        channel_embeddings: dict,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Args:
            channel_embeddings: Dict mapping channel name to embedding tensor.
            labels: Optional labels array aligned with embeddings.
        """
        self.channel_embeddings = channel_embeddings
        self.labels = labels

        # Validate alignment
        lengths = [len(v) for v in channel_embeddings.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Channel lengths don't match: {dict(zip(channel_embeddings.keys(), lengths))}")

        # Concatenate for main embeddings
        self.data = torch.cat(list(channel_embeddings.values()), dim=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return sample with concatenated embeddings and individual channels."""
        sample = {
            "embeddings": self.data[idx],
            "data": self.data[idx],
        }

        # Include individual channels
        for name, emb in self.channel_embeddings.items():
            sample[name] = emb[idx]

        # Include label if available
        if self.labels is not None:
            sample["label"] = self.labels[idx]

        return sample

    def get_labels(self):
        """Returns all labels for compatibility with plotting callbacks."""
        if self.labels is not None:
            return self.labels
        return np.zeros(len(self.data))
