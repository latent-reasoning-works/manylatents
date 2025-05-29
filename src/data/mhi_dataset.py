import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.data import load_metadata

from .plink_dataset import PlinkDataset
from .precomputed_mixin import PrecomputedMixin

logger = logging.getLogger(__name__)


class MHIDataset(PlinkDataset, PrecomputedMixin):
    """
    PyTorch Dataset for MHI data.
    Returns both raw data and (optionally) precomputed embeddings.
    """
    def __init__(self, 
                 files: Dict[str, str], 
                 cache_dir: str, 
                 data_split: str = "full",
                 mmap_mode: Optional[str] = None, 
                 precomputed_path: Optional[str] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 delimiter: Optional[str] = ",",
                 filter_qc: Optional[bool] = False,
                 filter_related: Optional[bool] = False,
                 test_all: Optional[bool] = False,
                 remove_recent_migration: Optional[bool] = False):
        """
        Initializes the MHI dataset.
        """
        self.data_split = data_split        
        self.filter_related = filter_related

        # Load raw data and metadata via the parent class.
        super().__init__(files=files, 
                         cache_dir=cache_dir, 
                         mmap_mode=mmap_mode,
                         delimiter=delimiter,
                         data_split=data_split,
                         precomputed_path=precomputed_path,
                         filter_qc=filter_qc,
                         filter_related=filter_related,
                         test_all=test_all,
                         remove_recent_migration=remove_recent_migration)

    def extract_geographic_preservation_indices(self) -> np.ndarray:
        """
        Extracts indices of samples that we expect to preserve geography.
        Returns:
            np.ndarray: Indices for subsetting for geographic preservation metric.        
        """

        return None
    
    def extract_latitude(self) -> pd.Series:
        """
        Extracts latitudes
        """
        if "latitude" not in self.metadata.columns:
            logger.warning("Latitude column not found in metadata. Returning zeros.")
            return pd.Series(np.zeros(len(self.metadata)), index=self.metadata.index)
        return self.metadata["latitude"]
    
    def extract_longitude(self) -> pd.Series:
        """
        Extracts longitudes
        """
        if "longitudes" not in self.metadata.columns:
            logger.warning("longitudes column not found in metadata. Returning zeros.")
            return pd.Series(np.zeros(len(self.metadata)), index=self.metadata.index)
        return self.metadata["longitudes"]


    def extract_population_label(self) -> pd.Series:
        """
        Extracts population labels
        """
        return self.metadata["label"]
    
    def extract_qc_filter_indices(self) -> np.ndarray:
        """
        Extracts points that has population labels.
        """
        filters = self.metadata["label"] == "Unlabelled"
        _filtered_indices = self.metadata[filters].index
        return ~self.metadata.index.isin(_filtered_indices)

    def extract_related_indices(self) -> np.ndarray:
        """
        Extracts maximal unrelated subset
        """
        return ~self.metadata['related'].values

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads and processes metadata for the HGDP dataset.
        """
        full_path = os.path.abspath(metadata_path)
        logger.info(f"Loading metadata from: {full_path}")

        # Define required columns.
        required_columns = [
            'sample_id',
            'label', 
            'related',
        ]

        metadata = load_metadata(
            file_path=full_path,
            required_columns=required_columns,
            additional_processing=None,
            delimiter=self.delimiter
        )

        # Check if the index has the required name; if not, try to set it.
        if metadata.index.name is None or metadata.index.name.strip() != 'sample_id':
            if 'sample_id' in metadata.columns:
                metadata = metadata.set_index('sample_id')
            else:
                raise ValueError("Missing required column: 'sample_id' in metadata.")

        # Convert filter columns to bool.
        filter_columns = ["related"]
        for col in filter_columns:
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(bool)
            else:
                logger.warning(f"Missing filter column in metadata: {col}. Filling with False.")
                metadata[col] = False

        return metadata

    def load_admixture_ratios(self, admixture_path, admixture_Ks) -> dict:
        """
        Loads admixture ratios
        """
        admixture_ratio_dict = super().load_admixture_ratios(admixture_path, admixture_Ks)
        return admixture_ratio_dict

    def get_labels(self, label_col: str = "label") -> np.ndarray:
        """
        Returns label array (e.g., Population) for coloring plots.
        """
        if label_col not in self.metadata.columns:
            raise ValueError(f"Label column '{label_col}' not found in metadata.")
        
        return self.metadata[label_col].values
    
    def extract_indices(self, 
                        filter_qc: bool,
                        filter_related: bool,
                        test_all: bool,
                        remove_recent_migration: bool
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts fit/transform indices based on metadata filters.
        Args:
            filter_qc (Optional[bool]): Whether to filter samples based on quality control.
            filter_related (Optional[bool]): Whether to filter related samples.
            test_all (Optional[bool]): Whether to use all samples for testing.
            remove_recent_migration (Optional[bool]): remove recently migrated samples.
        """
        fit_idx, trans_idx = super().extract_indices(filter_qc, filter_related, test_all, remove_recent_migration)

        # First entry is dummy row. So we ignore this!
        fit_idx[0] = False
        trans_idx[0] = False

        return fit_idx, trans_idx