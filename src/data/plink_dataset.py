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
                 precomputed_path: Optional[str] = None,
                 delimiter: Optional[str] = ",",
                 filter_qc: Optional[bool] = False,
                 filter_related: Optional[bool] = False,
                 test_all: Optional[bool] = False,
                 remove_recent_migration: Optional[bool] = False,
                 data_split: str = None,
                 ) -> None:
        """
        Initializes the PLINK dataset.

        Args:
            files (dict): Dictionary containing paths for PLINK and metadata files.
            cache_dir (str): Directory for caching preprocessed data.
            mmap_mode (Optional[str]): Memory-mapping mode for large datasets.
            precomputed_path (Optional[str]): path to precomputed embeddings.
            delimiter (Optional[str]): Delimiter for reading metadata files.
            filter_qc (Optional[bool]): Whether to filter samples based on quality control.
            filter_related (Optional[bool]): Whether to filter related samples.
            test_all (Optional[bool]): Whether to use all samples for testing.
            remove_recent_migration (Optional[bool]): remove recently migrated samples.
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
        self.admixture_path = files['admixture']
        self.admixture_ks = files['admixture_K'].split(',') if len(files['admixture_K']) > 0 else None

        self.metadata = self.load_metadata(self.metadata_path)
        self.admixture_ratios = self.load_admixture_ratios(self.admixture_path, self.admixture_ks)

        # get properties
        self._geographic_preservation_indices = self.extract_geographic_preservation_indices()
        self._latitude = self.extract_latitude()
        self._longitude = self.extract_longitude()
        self._population_label = self.extract_population_label()
        self._qc_filter_indices = self.extract_qc_filter_indices()
        self._related_indices = self.extract_related_indices()

        self.fit_idx, self.trans_idx = self.extract_indices(filter_qc,
                                                            filter_related,
                                                            test_all,
                                                            remove_recent_migration)

        self.split_indices = {
            'train': np.where(self.fit_idx)[0],
            'test': np.where(self.trans_idx)[0],
            'full': np.arange(len(self.metadata))
        }

        self.original_data = self.load_or_convert_data()
        
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
            
            # update exposed attributes
            self._latitude = self._latitude.iloc[idx].copy()
            self._longitude = self._longitude.iloc[idx].copy()
            self._population_label = self._population_label.iloc[idx].copy()
            self._qc_filter_indices = self._qc_filter_indices[idx]
            self._related_indices = self._related_indices[idx]
            self._geographic_preservation_indices = self._geographic_preservation_indices[idx]

            for K in self.admixture_ratios.keys():
                self.admixture_ratios[K] = self.admixture_ratios[K].iloc[idx].copy()

            # Update split_indices to an identity mapping.
            self.split_indices = {self.data_split: np.arange(len(self.metadata))}

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
        if filter_qc:
            filtered_indices = self.qc_filter_indices
        else:
            filtered_indices = np.ones(len(self.metadata), dtype=bool)

        if filter_related:
            related_indices = self.related_indices
        else:
            related_indices = np.ones(len(self.metadata), dtype=bool)
        
        if remove_recent_migration:
            recent_migrant_filter = self.geographic_preservation_indices
        else:
            recent_migrant_filter = np.ones(len(self.metadata), dtype=bool)

        if test_all:
            # for test set, include both related and unrelated
            fit_idx = related_indices & filtered_indices & recent_migrant_filter
            trans_idx = filtered_indices & recent_migrant_filter
        else:
            # otherwise train on unrelated and test on the related individuals
            fit_idx = related_indices & filtered_indices & recent_migrant_filter
            trans_idx = (~related_indices) & filtered_indices & recent_migrant_filter

        return fit_idx, trans_idx

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
    
    def load_admixture_ratios(self, admixture_path, admixture_Ks) -> dict:
        """
        Loads admixture ratios
        """
        admixture_ratio_dict = {}
        if admixture_Ks is not None:
            list_of_files = [admixture_path.replace('{K}', K) for K in admixture_Ks]
            for file, K in zip(list_of_files, admixture_Ks):
                admixture_ratio_dict[K] = pd.read_csv(file, sep='\t', header=None)
        return admixture_ratio_dict

    def __len__(self) -> int:
        return len(self.split_indices[self.data_split])

    def __getitem__(self, index: int) -> Any:
        real_idx = self.split_indices[self.data_split][index]
        sample = self.original_data[real_idx]
        metadata_row = self.metadata.iloc[real_idx].to_dict()
        metadata_row = {k.strip(): v for k, v in metadata_row.items()}
        return sample, metadata_row  

    @abstractmethod
    def extract_latitude(self) -> pd.Series:
        """
        Extracts latitudes
        """
        pass

    @abstractmethod
    def extract_longitude(self) -> pd.Series:
        """
        Extracts longitudes
        """
        pass

    @abstractmethod
    def extract_population_label(self) -> pd.Series:
        """
        Extracts population labels
        """
        pass

    @abstractmethod
    def extract_qc_filter_indices(self) -> np.ndarray:
        """
        Extracts points that passed QC
        """
        pass

    @abstractmethod
    def extract_related_indices(self) -> np.ndarray:
        """
        Extracts maximal unrelated subset
        """
        pass

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

    @property
    def get_original_data(self) -> np.ndarray:
        """
        Returns the original, unbatched data.
        """
        return self.original_data

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
    def qc_filter_indices(self) -> np.array:
        return self._qc_filter_indices

    @property
    def related_indices(self) -> np.array:
        return self._related_indices
    
    @property
    def geographic_preservation_indices(self) -> pd.Series:
        return self._geographic_preservation_indices
