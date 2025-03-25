import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.utils.data import (
    convert_plink_to_npy,
    generate_hash,
)

logger = logging.getLogger(__name__)

class PlinkDataset(Dataset):
    """
    PyTorch Dataset for PLINK-formatted genetic datasets.
    """
    
    _valid_splits = {"train", "test", "full"}

    def __init__(self, 
                 files: Dict[str, str], 
                 cache_dir: str,  
                 mmap_mode: Optional[str] = None,
                 delimiter: Optional[str] = ",",
                 filter_qc: Optional[bool] = False,
                 filter_related: Optional[bool] = False,
                 test_all: Optional[bool] = False,
                 data_split: str = None) -> None:
        """
        Initializes the PLINK dataset.

        Args:
            files (dict): Dictionary containing paths for PLINK and metadata files.
            cache_dir (str): Directory for caching preprocessed data.
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
            delimiter (Optional[str]): Delimiter for reading metadata files.
            filter_qc (Optional[bool]): Whether to filter samples based on quality control.
            filter_related (Optional[bool]): Whether to filter related samples.
            test_all (Optional[bool]): Whether to use all samples for testing.
            data_split (str): Data split to use ('train', 'test', or 'full').
        """
        super().__init__()
        
        if data_split not in self._valid_splits:
            raise ValueError(f"Invalid data_split '{data_split}'. Use one of {self._valid_splits}.")
        self.data_split = data_split
        self.filenames = files
        self.cache_dir = cache_dir 
        self.plink_path = files["plink"]
        self.metadata_path = files["metadata"]
        self.mmap_mode = mmap_mode
        self.delimiter = delimiter

        self.metadata = self.load_metadata(self.metadata_path)

        self.fit_idx, self.trans_idx = self.extract_indices()

        self.split_indices = {
            'train': np.where(self.fit_idx)[0],
            'test': np.where(self.trans_idx)[0],
            'full': np.arange(len(self.metadata))
        }

        self.original_data = self.load_or_convert_data()

    def load_or_convert_data(self) -> np.ndarray:
        """
        Loads or converts PLINK data to numpy format.
        """
        file_hash = generate_hash(self.plink_path, self.fit_idx, self.trans_idx)
        npy_cache_file = os.path.join(self.cache_dir, f".{file_hash}.npy")

        if not os.path.exists(npy_cache_file):
            logger.info("Converting PLINK data to numpy format...")
            convert_plink_to_npy(self.plink_path, npy_cache_file, self.fit_idx, self.trans_idx)

        logger.info(f"Loading processed PLINK data from {npy_cache_file}")
        return np.load(npy_cache_file, mmap_mode=self.mmap_mode)

    def __len__(self) -> int:
        return len(self.split_indices[self.data_split])

    def __getitem__(self, index: int) -> Any:
        real_idx = self.split_indices[self.data_split][index]
        sample = self.original_data[real_idx]
        metadata_row = self.metadata.iloc[real_idx].to_dict()
        metadata_row = {k.strip(): v for k, v in metadata_row.items()}
        return sample, metadata_row  
    
    @abstractmethod
    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets indices to fit and transform on using metadata.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Boolean arrays for fit and transform indices.
        """
        pass

    @property
    def get_original_data(self) -> np.ndarray:
        """
        Returns the original, unbatched data.
        """
        return self.original_data
    
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads metadata.

        Args:
            metadata_path (str): Path to the metadata file.

        Returns:
            pd.DataFrame: Loaded metadata DataFrame.
        """
        logger.info(f"Loading metadata from: {metadata_path}")
        return pd.read_csv(metadata_path, delimiter=self.delimiter)

    @abstractmethod
    def get_labels(self, label_col: str = "Population") -> np.ndarray:
        """
        Abstract method that should return an array of labels for the dataset.
        
        Args:
            label_col (str): Name of the column to use as labels.
        
        Returns:
            np.ndarray: Array of labels.
        """
        pass
