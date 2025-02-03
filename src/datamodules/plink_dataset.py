import hashlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from pyplink import PyPlink
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def replace_negative_one_with_nan(array: np.array) -> np.array:
    """
    Replace all occurrences of -1 with np.nan in the genotype array.

    Args:
        array (np.array): Genotype array.

    Returns:
        np.array: Modified array with -1 replaced by np.nan.
    """
    return np.where(array == -1, np.nan, array)

def hwe_normalize(genotypes_array: np.array, fit_idx: np.array) -> np.array:
    """
    Performs Hardy-Weinberg Equilibrium normalization on the genotype array.

    Args:
        genotypes_array (np.array): Genotype array.
        fit_idx (np.array): Boolean array indicating samples to fit on.

    Returns:
        np.array: HWE-normalized genotype array.
    """
    allele_freqs = np.nanmean(genotypes_array[fit_idx] / 2, axis=0)  # p = mean allele frequency
    centered_matrix = genotypes_array - 2 * allele_freqs
    hwe_variance = 2 * allele_freqs * (1 - allele_freqs)
    hwe_variance[hwe_variance == 0] = 1  # Avoid division by zero for monomorphic SNPs
    normalized_matrix = centered_matrix / np.sqrt(hwe_variance)
    return normalized_matrix

def preprocess_data_matrix(genotypes_array: np.array, fit_idx: np.array, trans_idx: np.array) -> np.array:
    """
    Preprocesses the genotype array by replacing -1 with NaN, performing HWE normalization,
    and handling missing values.

    Args:
        genotypes_array (np.array): Original genotype array.
        fit_idx (np.array): Boolean array indicating samples to fit on.
        trans_idx (np.array): Boolean array indicating samples to transform on.

    Returns:
        np.array: Preprocessed genotype array.
    """
    genotypes_array = replace_negative_one_with_nan(genotypes_array)
    normalized_matrix = hwe_normalize(genotypes_array, fit_idx)
    non_missing_mask = ~np.isnan(genotypes_array)
    normalized_matrix = np.where(non_missing_mask, normalized_matrix, 0)
    return normalized_matrix

def generate_hash(file_path: str, fit_idx: np.ndarray, trans_idx: np.ndarray) -> str:
    """
    Generate a unique hash based on the file path, last modified time, and two NumPy arrays.

    Args:
        file_path (str): Path to the file.
        fit_idx (np.ndarray): NumPy array (boolean).
        trans_idx (np.ndarray): NumPy array (boolean).

    Returns:
        str: A unique hash string.
    """
    # Check if required files exist
    if (
        os.path.exists(file_path + '.bed') and
        os.path.exists(file_path + '.bim') and
        os.path.exists(file_path + '.fam')
    ):
        last_modified = os.path.getmtime(file_path + '.bed')
    else:
        raise FileNotFoundError(
            f"One or more required files are missing: {file_path}.bed, "
            f"{file_path}.bim, {file_path}.fam"
        )

    # Convert file metadata to hash input
    file_name = os.path.basename(file_path)
    hash_input = f"{file_name}:{last_modified}".encode("utf-8")

    # Include the contents of the NumPy arrays in the hash
    hash_input += fit_idx.tobytes()  # Convert fit_idx to bytes
    hash_input += trans_idx.tobytes()  # Convert trans_idx to bytes

    # Generate and return the hash
    return hashlib.md5(hash_input).hexdigest()

def convert_plink_to_npy(plink_prefix: str, fname: str, fit_idx: np.array, trans_idx: np.array) -> None:
    """
    Converts PLINK binary files to a NumPy array after preprocessing.

    Args:
        plink_prefix (str): Prefix for the PLINK binary files.
        fname (str): Filename for the output NumPy array.
        fit_idx (np.array): Boolean array indicating samples to fit on.
        trans_idx (np.array): Boolean array indicating samples to transform on.
    """
    pedfile = PyPlink(plink_prefix)
    genotypes_array = np.zeros([pedfile.get_nb_samples(), pedfile.get_nb_markers()], dtype=np.int8)

    for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile), total=pedfile.get_nb_markers()):
        genotypes_array[:, i] = genotypes

    # HWE normalization
    normalized_matrix = preprocess_data_matrix(genotypes_array, fit_idx, trans_idx)

    np.save(fname, normalized_matrix)

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
