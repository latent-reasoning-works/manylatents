import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.data import load_metadata

from .plink_dataset import PlinkDataset
from .precomputed_mixin import PrecomputedMixin

logger = logging.getLogger(__name__)


def hgdp_add_dummy_row(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a dummy row to the metadata DataFrame to account for missing data in the first row.
    """
    null_row = pd.DataFrame([{col: np.nan for col in metadata.columns}])
    filter_columns = ["filter_king_related", "filter_pca_outlier", "hard_filtered", "filter_contaminated"]
    for _filter in filter_columns:
        if _filter in metadata.columns:
            null_row[_filter] = False
    metadata = pd.concat([null_row, metadata], ignore_index=True)
    return metadata


class HGDPDataset(PlinkDataset, PrecomputedMixin):
    """
    PyTorch Dataset for HGDP + 1000 Genomes data.
    Returns both raw data and (optionally) precomputed embeddings.
    """
    def __init__(self, 
                 files: Dict[str, str], 
                 cache_dir: str, 
                 filter_related: bool = True,
                 data_split: str = "full",
                 mmap_mode: Optional[str] = None, 
                 precomputed_path: Optional[str] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 delimiter: Optional[str] = ","):
        """
        Initializes the HGDP dataset.
        """
        self.data_split = data_split        
        self.filter_related = filter_related

        # Load raw data and metadata via the parent class.
        super().__init__(files=files, 
                         cache_dir=cache_dir, 
                         mmap_mode=mmap_mode,
                         delimiter=delimiter,
                         data_split=data_split)
        
        # Load precomputed embeddings using the mixin, if provided.
        self.precomputed_path = precomputed_path
        self.precomputed_embeddings = self.load_precomputed(precomputed_path, mmap_mode=mmap_mode)
        
        # Note: Do NOT override self.original_data here,
        # so that raw data remains available for evaluations.
        if self.data_split != "full":
            idx = self.split_indices[self.data_split]
            self.metadata = self.metadata.iloc[idx].copy()
            self.original_data = self.original_data[idx]
            if self.precomputed_embeddings is not None:
                self.precomputed_embeddings = self.precomputed_embeddings[idx]
            # Update split_indices to an identity mapping.
            self.split_indices = {self.data_split: np.arange(len(self.metadata))}

        # get properties
        self._geographic_preservation_indices = self.extract_geographic_preservation_indices()
        self._latitude = self.metadata["latitude"]
        self._longitude = self.metadata["longitude"]
        self._population_label = self.metadata["Population"]

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

    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts fit/transform indices based on metadata filters.
        """
        filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        _filtered_indices = self.metadata[self.metadata[filters].any(axis=1)].index
        filtered_indices = ~self.metadata.index.isin(_filtered_indices)
        
        if self.filter_related:
            related_indices = ~self.metadata['filter_king_related'].values
        else:
            related_indices = np.ones(len(self.metadata), dtype=bool)

        fit_idx = related_indices & filtered_indices
        trans_idx = (~related_indices) & filtered_indices

        return fit_idx, trans_idx

    def extract_geographic_preservation_indices(self) -> np.ndarray:
        """
        Extracts indices of samples that we expect to preserve geography.
        Returns:
            np.ndarray: Indices for subsetting for geographic preservation metric.        
        """

        american_idx = self.metadata['Genetic_region_merged'] == 'America'
        rest_idx = self.metadata['Population'].isin(['ACB', 'ASW', 'CEU'])

        return ~(american_idx | rest_idx)

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Loads and processes metadata for the HGDP dataset.
        """
        full_path = os.path.abspath(metadata_path)
        logger.info(f"Loading metadata from: {full_path}")

        # Define required columns.
        required_columns = [
            'project_meta.sample_id',
            'filter_king_related',
            'filter_pca_outlier',
            'hard_filtered',
            'filter_contaminated',
            'Genetic_region_merged',
            'Population'
        ]

        metadata = load_metadata(
            file_path=full_path,
            required_columns=required_columns,
            additional_processing=hgdp_add_dummy_row,
            delimiter=self.delimiter
        )

        # Check if the index has the required name; if not, try to set it.
        if metadata.index.name is None or metadata.index.name.strip() != 'project_meta.sample_id':
            if 'project_meta.sample_id' in metadata.columns:
                metadata = metadata.set_index('project_meta.sample_id')
            else:
                raise ValueError("Missing required column: 'project_meta.sample_id' in metadata.")

        # Convert filter columns to bool.
        filter_columns = ["filter_king_related", "filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        for col in filter_columns:
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(bool)
            else:
                logger.warning(f"Missing filter column in metadata: {col}. Filling with False.")
                metadata[col] = False

        return metadata

    def get_labels(self, label_col: str = "Population") -> np.ndarray:
        """
        Returns label array (e.g., Population) for coloring plots.
        """
        if label_col not in self.metadata.columns:
            raise ValueError(f"Label column '{label_col}' not found in metadata.")
        
        return self.metadata[label_col].values
   
    @property
    def latitude(self) -> pd.Series:
        return self._latitude

    @property
    def longitude(self) -> pd.Series:
        return self._longitude

    @property
    def population_label(self) -> pd.Series:
        return self._population_label
    
    @property
    def geographic_preservation_indices(self) -> pd.Series:
        return self._geographic_preservation_indices
