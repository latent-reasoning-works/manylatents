import logging
import os
from typing import Union, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from manylatents.utils.data import load_metadata

from .plink_dataset import PlinkDataset
from .precomputed_mixin import PrecomputedMixin

logger = logging.getLogger(__name__)


class AOUDataset(PlinkDataset, PrecomputedMixin):
    """
    PyTorch Dataset for AoU data.
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
                 balance_filter: Union[bool, float] = False,
                 include_do_not_know: bool = False,
                 test_all: Optional[bool] = False):
        """
        Initializes the AOU dataset.
        """
        self.data_split = data_split        
        self.filter_related = filter_related
        self.include_do_not_know  = include_do_not_know

        # Load raw data and metadata via the parent class.
        super().__init__(files=files, 
                         cache_dir=cache_dir, 
                         mmap_mode=mmap_mode,
                         delimiter=delimiter,
                         data_split=data_split,
                         precomputed_path=precomputed_path,
                         filter_qc=filter_qc,
                         filter_related=filter_related,
                         balance_filter=balance_filter,
                         test_all=test_all)

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
            return pd.Series(np.zeros(len(self.metadata)), 
                             index=self.metadata.index)
        return self.metadata["latitude"]
    
    def extract_longitude(self) -> pd.Series:
        """
        Extracts longitudes
        """
        if "longitudes" not in self.metadata.columns:
            logger.warning("longitudes column not found in metadata. Returning zeros.")
            return pd.Series(np.zeros(len(self.metadata)), 
                             index=self.metadata.index)
        return self.metadata["longitudes"]
    
    def extract_population_label(self) -> pd.Series:
        """
        Extracts population labels
        """
        return self.metadata["SelfReportedRaceEthnicity"]

    def extract_qc_filter_indices(self) -> np.ndarray:
        """
        Extracts points that passed QC
        """
        #filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        #_filtered_indices = self.metadata[self.metadata[filters].any(axis=1)].index
        #return ~self.metadata.index.isin(_filtered_indices)
        # Currently all points pass QC
        return np.ones(len(self.metadata), dtype=bool)

    def extract_related_indices(self) -> np.ndarray:
        """
        Extracts maximal unrelated subset
        """
        return np.ones(len(self.metadata), dtype=bool)

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads and processes metadata for the UKBB dataset.
        """
        full_path = os.path.abspath(metadata_path)
        logger.info(f"Loading metadata from: {full_path}")

        # Define required columns.
        required_columns = ['fid',
                            'sample_id',
                            'person_id',
                            'SelfReportedRaceEthnicity',
                            'date_of_birth',
                            'race',
                            'ethnicity',
                            'sex_at_birth'
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
        filter_columns = ["filter_related"]
        for col in filter_columns:
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(bool)
            else:
                logger.warning(f"Missing filter column in metadata: {col}. Filling with False.")
                metadata[col] = False

        return metadata

    def get_labels(self, label_col: str = "SelfReportedRaceEthnicity") -> np.ndarray:
        """
        Returns label array (e.g., SelfReportedRaceEthnicity) for coloring plots.
        """
        if label_col not in self.metadata.columns:
            raise ValueError(f"Label column '{label_col}' not found in metadata.")

        return self.metadata[label_col].values
    
    def balance_filter(self, balance_filter) -> np.array:

        num_dominant = self.metadata[(self.metadata['SelfReportedRaceEthnicity'] == 'White')].shape[0]

        # "Do not know" enriched for EUR. So treat them as dominant
        num_nondominant = self.metadata[(self.metadata['SelfReportedRaceEthnicity'] != 'White') & (self.metadata['SelfReportedRaceEthnicity'] != 'No information')].shape[0]

        num_to_subset = int(num_nondominant * balance_filter)

        np.random.seed(42)  # Ensure reproducible subsampling
        EUR_subset = np.random.choice(self.metadata[self.metadata['SelfReportedRaceEthnicity'] == 'White'].person_id,
                             num_to_subset,
                             replace=False)
        #print(f'subsetting EUR from {num_dominant} to {num_to_subset}')
        if self.include_do_not_know:
            rest = self.metadata[self.metadata['SelfReportedRaceEthnicity'] != 'White'].person_id
        else:
            rest = self.metadata[(self.metadata['SelfReportedRaceEthnicity'] != 'White') & (self.metadata['SelfReportedRaceEthnicity'] != 'No information')].person_id

        idxs = np.concatenate([EUR_subset, rest])
        boolean_idx = self.metadata.person_id.isin(idxs)

        return boolean_idx
