import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.data import load_metadata

from .plink_dataset import PlinkDataset
from .precomputed_mixin import PrecomputedMixin

logger = logging.getLogger(__name__)


class UKBBDataset(PlinkDataset, PrecomputedMixin):
    """
    PyTorch Dataset for UKBB data.
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
        Initializes the UKBB dataset.
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

    def __getitem__(self, index: int) -> Any:
        real_idx = self.split_indices[self.data_split][index]
        sample_raw = self.original_data[real_idx]
        sample_precomputed = None
        if self.precomputed_embeddings is not None:
            sample_precomputed = self.precomputed_embeddings[real_idx]
        
        metadata_row = self.metadata.iloc[real_idx].to_dict()
        metadata_row = {k.strip(): v for k, v in metadata_row.items()}
        
        # Return a dict containing both raw and precomputed data.
        return {
            "raw": sample_raw,
            "precomputed": sample_precomputed,
            "metadata": metadata_row
        }

    def extract_geographic_preservation_indices(self) -> np.ndarray:
        """
        Extracts indices of samples that we expect to preserve geography.
        Returns:
            np.ndarray: Indices for subsetting for geographic preservation metric.        
        """

        # Only look at British
        gbr_idx = self.metadata['self_described_ancestry'].isin(['British', 'Irish'])

        # Only look at rural areas
        high_density_idx = self.metadata['urban / rural'].isin(['England/Wales - Urban - less sparse',
                                                                'Scotland - Large Urban Area',
                                                                'Scotland - Other Urban Area',
                                                                'England/Wales - Urban - sparse'])

        # Make sure these have longitude/latitude information
        has_long_lat = ~self.metadata['latitude'].isna() & ~self.metadata['longitude'].isna()
        return (gbr_idx & ~high_density_idx & has_long_lat)

    def extract_latitude(self) -> pd.Series:
        """
        Extracts latitudes
        """
        return self.metadata["latitude"]
    
    def extract_longitude(self) -> pd.Series:
        """
        Extracts longitudes
        """
        return self.metadata["longitude"]
    
    def extract_population_label(self) -> pd.Series:
        """
        Extracts population labels
        """
        return self.metadata["self_described_ancestry"]
    
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
        return self.metadata['filter_related'].values

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads and processes metadata for the UKBB dataset.
        """
        full_path = os.path.abspath(metadata_path)
        logger.info(f"Loading metadata from: {full_path}")

        # Define required columns.
        required_columns = ['IDs',
                            'self_described_ancestry',
                            'latitude',
                            'longitude',
                            'age',
                            'filter_related',
                            'urban / rural'
                           ]

        metadata = load_metadata(
            file_path=full_path,
            required_columns=required_columns,
            additional_processing=None,
            delimiter=self.delimiter
        )

        # Check if the index has the required name; if not, try to set it.
        if metadata.index.name is None or metadata.index.name.strip() != 'IDs':
            if 'IDs' in metadata.columns:
                metadata = metadata.set_index('IDs')
            else:
                raise ValueError("Missing required column: 'IDs' in metadata.")

        # Convert filter columns to bool.
        filter_columns = ["filter_related"]
        for col in filter_columns:
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(bool)
            else:
                logger.warning(f"Missing filter column in metadata: {col}. Filling with False.")
                metadata[col] = False

        return metadata

    def get_labels(self, label_col: str = "self_described_ancestry") -> np.ndarray:
        """
        Returns label array (e.g., self_described_ancestry) for coloring plots.
        """
        if label_col not in self.metadata.columns:
            raise ValueError(f"Label column '{label_col}' not found in metadata.")
        
        return self.metadata[label_col].values