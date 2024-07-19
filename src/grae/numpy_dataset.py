import torch
from torch.utils.data import Dataset

class FromNumpyDataset(Dataset):
    """Torch Dataset Wrapper for x ndarray with no target."""

    def __init__(self, x):
        """Create torch wraper dataset form simple ndarray.
        Args:
            x (ndarray): Input variables.
        """
        self._data = torch.from_numpy(x).float()

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def numpy(self, idx=None):
        """Get dataset as ndarray.
        Specify indices to return a subset of the dataset, otherwise return whole dataset.
        Args:
            idx (int, optional): Specify index or indices to return.
        Returns:
            ndarray: Return flattened dataset as a ndarray.
        """
        n = len(self)

        data = self._data.numpy().reshape((n, -1))

        if idx is None:
            return data
        else:
            return data[idx]