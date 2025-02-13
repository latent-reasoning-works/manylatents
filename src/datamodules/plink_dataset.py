import logging
import os
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

    def __init__(self, 
                 files: Dict[str, str], 
                 cache_dir: str,  
                 mmap_mode: Optional[str] = None,) -> None:
        """
        Initializes the PLINK dataset.

        Args:
            filenames (dict): Dictionary containing paths for PLINK and metadata files.
            cache_dir (str): Directory for caching preprocessed data.
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
            mode (str): Determines type of data returned ('genotypes' or 'pca').
        """
        super().__init__()
        self.filenames = files
        self.cache_dir = cache_dir 
        self.plink_path = files["plink"]
        self.metadata_path = files["metadata"]
        self.mmap_mode = mmap_mode

        # Load metadata
        self.metadata = self.load_metadata(self.metadata_path)

        # Extract fit and transform indices
        ## these will go into the dataloader
        self.fit_idx, self.trans_idx = self.extract_indices()

        # Generate unique cache file paths
        file_hash = generate_hash(self.plink_path, self.fit_idx, self.trans_idx)
        self.npy_cache_file = os.path.join(self.cache_dir, f".{file_hash}.npy")
        self.pca_cache_file = os.path.join(self.cache_dir, f".{file_hash}.pca.npy")

        if not os.path.exists(self.npy_cache_file):
            convert_plink_to_npy(self.plink_path, self.npy_cache_file, self.fit_idx, self.trans_idx)
            ## this creates the X matrix
        self.X = np.load(self.npy_cache_file, mmap_mode=self.mmap_mode)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and metadata data, optionally transformed by the respective transforms.
        """
        return self.X[index], self.metadata.iloc[index]

    def __len__(self) -> int:
        return len(self.X)
    
    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sets indices to fit and transform on using metadata.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Boolean arrays for fit and transform indices.
        """
        raise NotImplementedError
    
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads metadata.

        Args:
            metadata_path (str): Path to the metadata file.

        Returns:
            pd.DataFrame: Loaded metadata DataFrame.
        """
        raise NotImplementedError
