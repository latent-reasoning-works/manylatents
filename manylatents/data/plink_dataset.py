import logging
import os
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from manylatents.utils.data import (
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
                 cache_dir: str,  
                 files: Dict[str, str] = None, 
                 metadata: Optional[pd.DataFrame] = None,
                 mmap_mode: Optional[str] = None,
                 precomputed_path: Optional[str] = None,
                 delimiter: Optional[str] = ",",
                 filter_qc: Optional[bool] = False,
                 filter_related: Optional[bool] = False,
                 balance_filter: Union[bool, float] = False,
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
            balance_filter (Union[bool, float]): subset the predominant class to be this percent of the dataset.
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
        self.mmap_mode = mmap_mode
        self.delimiter = delimiter
        self.admixture_path = files.get('admixture', None)
        admixture_K_str = files.get('admixture_K', '')
        self.admixture_ks = admixture_K_str.split(',') if len(admixture_K_str) > 0 else None

        # Step 1: Load metadata
        if files is not None and "plink" in files:
            self.plink_path = files["plink"]
            self.metadata_path = files["metadata"]
            if self.metadata_path is None:
                raise ValueError("Metadata path must be provided in the files dict.")
            self.metadata = self.load_metadata(self.metadata_path)
        elif files is not None and "metadata" in files:
            # Only metadata is provided; no raw data available.
            self.metadata = self.load_metadata(files["metadata"])
        elif metadata is not None:
            self.metadata = metadata
        else:
            raise ValueError("Must provide either a files dict or metadata directly.")

        # Step 2: Extract metadata-derived properties
        self._latitude = self.extract_latitude()
        self._longitude = self.extract_longitude()
        self._population_label = self.extract_population_label()
        self._qc_filter_indices = self.extract_qc_filter_indices()
        self._related_indices = self.extract_related_indices()
        self._geographic_preservation_indices = self.extract_geographic_preservation_indices()
        self.admixture_ratios = self.load_admixture_ratios(self.admixture_path, self.admixture_ks)


        # Step 3: Extract indices
        self.fit_idx, self.trans_idx = self.extract_indices(filter_qc,
                                                            filter_related,
                                                            test_all,
                                                            remove_recent_migration,
                                                            balance_filter)

        self.split_indices = {
            'train': np.where(self.fit_idx)[0],
            'test': np.where(self.trans_idx)[0],
            'full': np.arange(len(self.metadata))
        }

        # Step 4: Load raw or precomputed data
        _data = None
        if precomputed_path is not None:
            _data = self.load_precomputed(precomputed_path, mmap_mode=mmap_mode)
        else:
            _data = self.load_or_convert_data()
        if _data is None:
            raise ValueError("No data source found: either raw Plink or precomputed embeddings are missing.")  
        
        full_n = len(self.split_indices["full"])
        split_idx = self.split_indices[self.data_split]
        # Step 5: Slice data
        if _data.shape[0] == full_n: # We have the full matrix → use our helper to slice
            self.data = self._apply_split(_data, split=self.data_split)
        else: # Precomputed is already sliced to split size (e.g. 4094)
            if _data.shape[0] != len(split_idx):
                raise ValueError(
                    f"Precomputed embeddings have {_data.shape[0]} rows but split '{self.data_split}' has {len(split_idx)} samples"
                )
            # Manually slice metadata & all per-sample arrays in place
            self.metadata = self.metadata.iloc[split_idx].reset_index(drop=True)
            for name in [
                "_latitude", "_longitude", "_population_label",
                "_qc_filter_indices", "_related_indices", "_geographic_preservation_indices"
            ]:
                val = getattr(self, name)
                if val is None:
                    setattr(self, name, None)
                elif isinstance(val, pd.Series):
                    setattr(self, name, val.iloc[split_idx].reset_index(drop=True))
                else:
                    setattr(self, name, val[split_idx])
            # Slice admixture_ratios
            for K, df in self.admixture_ratios.items():
                self.admixture_ratios[K] = df.iloc[split_idx].reset_index(drop=True)
            self.data = _data

        assert self.data.shape[0] == len(self._latitude) == len(self.metadata), (
            f"Split mismatch: data has {self.data.shape[0]} rows, "
            f"latitude {len(self._latitude)}, metadata {len(self.metadata)}"
        )

    def _apply_split(self, raw: np.ndarray, split: str) -> np.ndarray:
        """
        Subset raw (samples×features) and all per-sample attrs to the given split.
        Returns the sliced raw; updates self.metadata and all self._* arrays in place.
        """
        if split == "full":
            return raw

        idx = self.split_indices[split]
        # 1) slice the raw matrix
        sliced = raw[idx, ...]

        # 2) slice every per-sample attribute in __dict__
        for name, val in list(self.__dict__.items()):
            # only our private per-sample arrays start with "_" 
            if not name.startswith("_"):
                continue
            # numpy array
            if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) == raw.shape[0]:
                setattr(self, name, val[idx])
            # pandas Series/DataFrame
            elif isinstance(val, (pd.Series, pd.DataFrame)) and len(val) == raw.shape[0]:
                setattr(self, name, val.iloc[idx].reset_index(drop=True))

        # 3) slice admixture_ratios dict
        for K, df in self.admixture_ratios.items():
            if isinstance(df, (pd.Series, pd.DataFrame)) and len(df) == raw.shape[0]:
                self.admixture_ratios[K] = df.iloc[idx].reset_index(drop=True)

        # 4) slice metadata itself
        self.metadata = self.metadata.iloc[idx].reset_index(drop=True)

        return sliced

    def extract_indices(self, 
                        filter_qc: bool,
                        filter_related: bool,
                        test_all: bool,
                        remove_recent_migration: bool,
                        balance_filter: Union[bool, float]
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts fit/transform indices based on metadata filters.
        Args:
            filter_qc (Optional[bool]): Whether to filter samples based on quality control.
            filter_related (Optional[bool]): Whether to filter related samples.
            test_all (Optional[bool]): Whether to use all samples for testing.
            remove_recent_migration (Optional[bool]): remove recently migrated samples.
            balance_filter (Union[bool, float]): subset the predominant class to be this percent of the dataset.
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
        
        if balance_filter:
            balanced_set = self.balance_filter(balance_filter)
        else:
            balanced_set = np.ones(len(self.metadata), dtype=bool)

        if test_all:
            # for test set, include both related and unrelated
            fit_idx = related_indices & filtered_indices & recent_migrant_filter & balanced_set
            #trans_idx = filtered_indices & recent_migrant_filter & balanced_set
            trans_idx = np.ones(len(self.metadata), dtype=bool)
        else:
            # otherwise train on unrelated and test on the related individuals
            fit_idx = related_indices & filtered_indices & recent_migrant_filter & balanced_set
            #trans_idx = (~related_indices) & filtered_indices & recent_migrant_filter & balanced_set
            trans_idx = filtered_indices & recent_migrant_filter & balanced_set
            
        logger.info(f"Fitting {fit_idx.sum()} points. Transforming {trans_idx.sum()} points")

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
                # only keep admixture info + sample IDs. Drop other columns
                admixture_ratio_dict[K] = pd.read_csv(file, sep='\t', header=None).iloc[:, :-2]
        return admixture_ratio_dict

    def __len__(self) -> int:
        return len(self.split_indices[self.data_split])

    def __getitem__(self, idx: int) -> Any:
        sample = self.data[idx]
        metadata_row = self.metadata.iloc[idx].to_dict()
        metadata_row = {k.strip(): v for k, v in metadata_row.items()}
        return {"data": sample, "metadata": metadata_row}

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

    @abstractmethod
    def get_sample_ids(self) -> np.ndarray:
        """
        Abstract method that should return an array of sample IDs for the dataset.

        Returns:
            np.ndarray: Array of sample IDs.
        """
        pass
    
    @abstractmethod
    def balance_filter(self, balance_filter) -> np.array:
        pass
    
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
