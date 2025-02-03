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
    PyTorch Dataset for the Thousand Genomes Project + Human Genome Diversity Project (HGDP) dataset.
    """

    def __init__(self, 
                 files: Dict[str, str], cache_dir: str, 
                 mode: str = 'genotypes', mmap_mode: Optional[str] = None):
        """
        Initializes the HGDP dataset with configuration parameters.

        Args:
            files (dict): Dictionary containing the file paths for plink and metadata.
            mode (str): Determines the type of data returned ('genotypes' or 'pca').
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
        """
        super().__init__(
            files=files,
            cache_dir=cache_dir,
            mmap_mode=mmap_mode, 
            mode=mode
        )

    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
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
        logger.info(f"Loading metadata from: {full_path}")  # Debugging

        # Define required columns
        required_columns = ['project_meta.sample_id',] 
                            #'filter_king_related', 
                            #'filter_pca_outlier', 
                            #'hard_filtered', 
                            #'filter_contaminated']

        metadata = load_metadata(
            file_path=full_path,
            required_columns=required_columns,
            additional_processing=hgdp_add_dummy_row
        )

        return metadata.set_index('project_meta.sample_id')
