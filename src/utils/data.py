import csv
import logging
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pyplink import PyPlink
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as svstack
import hashlib
import tqdm
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .mappings import make_palette_label_order_HGDP

class DummyDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def detect_separator(file_path: str, sample_size: int = 1024) -> Optional[str]:
    """
    Detects the delimiter of a file using csv.Sniffer.

    Args:
        file_path (str): Path to the file.
        sample_size (int): Number of bytes to read for detection.

    Returns:
        Optional[str]: Detected delimiter or None if detection fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(sample_size)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            logging.info(f"Detected delimiter '{delimiter}' for file '{file_path}'.")
            return delimiter
    except csv.Error as e:
        logging.warning(f"CSV parsing error while detecting delimiter for '{file_path}': {e}")
        return None
    except UnicodeDecodeError as e:
        logging.warning(f"Encoding error while reading '{file_path}': {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error while detecting delimiter for '{file_path}': {e}")
        return None

def load_metadata(
    file_path: str,
    required_columns: Optional[list] = None,
    additional_processing: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Loads metadata from a file, handling delimiter detection and basic validation.

    Args:
        file_path (str): Path to the metadata file.
        required_columns (Optional[list]): List of columns that must be present.
        additional_processing (Optional[Callable[[pd.DataFrame], pd.DataFrame]]): 
            A function to perform additional processing on the DataFrame.

    Returns:
        pd.DataFrame: Loaded and processed metadata.
    """
    logging.info(f"Loading metadata from: {file_path}")

    # Detect delimiter
    delimiter = detect_separator(file_path)

    # Fallback to tab if detection fails
    if delimiter is None:
        logging.warning(f"Falling back to tab separator for file: {file_path}")
        delimiter = '\t'

    try:
        metadata = pd.read_csv(file_path, sep=delimiter, engine='python')
        logging.info(f"Successfully loaded metadata with delimiter '{delimiter}'.")
    except pd.errors.ParserError as e:
        logging.error(f"ParserError while reading {file_path}: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading {file_path}: {e}")
        raise

    # Validate required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            logging.error(f"Missing required columns {missing_columns} in metadata file: {file_path}")
            raise KeyError(f"Missing required columns {missing_columns} in metadata file: {file_path}")

    # Perform additional processing if provided
    if additional_processing:
        metadata = additional_processing(metadata)

    return metadata

### Plink Dataset
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
### Plink Dataset end