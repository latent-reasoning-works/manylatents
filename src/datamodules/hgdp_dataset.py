import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.data import load_metadata

from .plink_dataset import PlinkDataset

logger = logging.getLogger(__name__)


def hgdp_add_dummy_row(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a dummy row to the metadata DataFrame to account for missing data in the first row.

    Args:
        metadata (pd.DataFrame): The original metadata DataFrame.

    Returns:
        pd.DataFrame: The modified metadata DataFrame with a dummy row.
    """
    null_row = pd.DataFrame([{col: np.nan for col in metadata.columns}])
    
    # Conditionally set filter columns if they exist
    filter_columns = ["filter_king_related", "filter_pca_outlier", "hard_filtered", "filter_contaminated"]
    for _filter in filter_columns:
        if _filter in metadata.columns:
            null_row[_filter] = False
    
    metadata = pd.concat([null_row, metadata], ignore_index=True)
    return metadata

class HGDPDataset(PlinkDataset):
    """
    PyTorch Dataset for HGDP + 1000 Genomes data.
    """
    def __init__(self, 
                 files: Dict[str, str], 
                 cache_dir: str, 
                 mmap_mode: Optional[str] = None, 
                 precomputed_path: Optional[str] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 delimiter: Optional[str] = ","):
        """
        Initializes the HGDP dataset.

        Args:
            files (dict): Paths for PLINK and metadata files.
            cache_dir (str): Directory for caching.
            mmap_mode (Optional[str]): Memory-mapping mode.
            precomputed_path (Optional[str]): Path to precomputed embeddings if available.
        """
        super().__init__(files=files, 
                         cache_dir=cache_dir, 
                         mmap_mode=mmap_mode,)

        self.precomputed_path = precomputed_path
        self.delimiter = delimiter
        self.metadata = metadata if metadata is not None else self.load_metadata(files["metadata"])
        
        # Load precomputed embeddings if provided
        if self.precomputed_path and os.path.exists(self.precomputed_path):
            logger.info(f"Loading precomputed embeddings from {self.precomputed_path}")
            if self.precomputed_path.endswith(".npy"):
                self.X = np.load(self.precomputed_path, mmap_mode=self.mmap_mode)
            elif self.precomputed_path.endswith(".csv"):
                self.X = np.loadtxt(self.precomputed_path, delimiter=",")
            else:
                raise ValueError(f"Unsupported file format: {self.precomputed_path}")

        # Extract indices based on metadata (even for precomputed)
        self.fit_idx, self.trans_idx = self.extract_indices()

    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts fit/transform indices based on metadata filters.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices for fitting and transforming.
        """
        filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        _filtered_indices = self.metadata[self.metadata[filters].any(axis=1)].index
        filtered_indices = ~self.metadata.index.isin(_filtered_indices)
        related_indices = ~self.metadata['filter_king_related'].values

        fit_idx = related_indices & filtered_indices
        trans_idx = (~related_indices) & filtered_indices

        return fit_idx, trans_idx

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads and processes metadata for the HGDP dataset.

        Args:
            metadata_path (str): Path to the metadata file.

        Returns:
            pd.DataFrame: Processed metadata DataFrame.
        """
        full_path = os.path.abspath(metadata_path)
        logger.info(f"Loading metadata from: {full_path}")

        # Define required columns
        required_columns = ['project_meta.sample_id', 
                            'filter_king_related', 
                            'filter_pca_outlier', 
                            'hard_filtered', 
                            'filter_contaminated']

        metadata = load_metadata(
            file_path=full_path,
            required_columns=required_columns,
            additional_processing=hgdp_add_dummy_row,
            delimiter=self.delimiter
        )

        return metadata.set_index('project_meta.sample_id')