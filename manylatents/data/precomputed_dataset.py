import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class PrecomputedDataset(Dataset):
    """
    A generic PyTorch Dataset for loading pre-computed embeddings from CSV or NPY files.
    """
    def __init__(self, path: str, label_col: str = None):
        """
        Args:
            path (str): Path to the data file (.csv or .npy).
            label_col (str, optional): The name of the column containing labels in a CSV file.
                                       If None, no labels are loaded.
        """
        self.path = path
        self.label_col = label_col
        self._load_data()

    def _load_data(self):
        logger.info(f"Loading pre-computed data from: {self.path}")
        if self.path.endswith('.csv'):
            df = pd.read_csv(self.path)
            if self.label_col and self.label_col in df.columns:
                self.labels = torch.tensor(pd.to_numeric(df[self.label_col], errors='coerce').fillna(0).values)
                self.data = torch.tensor(df.drop(columns=[self.label_col]).values, dtype=torch.float32)
            else:
                self.labels = torch.zeros(len(df), dtype=torch.long)
                self.data = torch.tensor(df.values, dtype=torch.float32)

        elif self.path.endswith('.npy'):
            self.data = torch.tensor(np.load(self.path), dtype=torch.float32)
            self.labels = torch.zeros(len(self.data), dtype=torch.long)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .npy.")
            
        logger.info(f"Successfully loaded data with shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The metadata value is a dummy label for compatibility with existing callbacks
        return {"data": self.data[idx], "metadata": self.labels[idx].item()}

    def get_labels(self):
        """Returns all labels for compatibility with plotting callbacks."""
        return self.labels.numpy()