import os
import hashlib
import tqdm
from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from pyplink import PyPlink
from torch.utils.data import Dataset


# TODO: clean this up
CACHE_DIR = '.'

def replace_negative_one_with_nan(array: np.array) -> np.array:
    # Replace all occurrences of -1 with np.nan
    return np.where(array == -1, np.nan, array)


def hwe_normalize(genotypes_array: np.array,
                  fit_idx: np.array) -> np.array:

    # Compute allele frequencies, ignoring NaNs
    allele_freqs = np.nanmean(genotypes_array[fit_idx] / 2, axis=0)  # p = mean allele frequency

    # Center the matrix by subtracting 2 * allele frequency for each SNP
    centered_matrix = genotypes_array - 2 * allele_freqs

    # Compute Hardy-Weinberg variance for each SNP, avoiding division by zero
    hwe_variance = 2 * allele_freqs * (1 - allele_freqs)
    hwe_variance[hwe_variance == 0] = 1  # Avoid division by zero for monomorphic SNPs

    # Normalize each SNP by Hardy-Weinberg variance
    normalized_matrix = centered_matrix / np.sqrt(hwe_variance)
    return normalized_matrix


def preprocess_data_matrix(genotypes_array: np.array,
                           fit_idx: np.array,
                           trans_idx: np.array) -> np.array:

    # Compute hwe normalized matrix
    genotypes_array = replace_negative_one_with_nan(genotypes_array)
    normalized_matrix = hwe_normalize(genotypes_array, fit_idx)

    # Create a mask for non-missing values
    non_missing_mask = ~np.isnan(genotypes_array)

    # Replace NaNs in the normalized matrix with zeros 
    # for compatibility with matrix multiplication
    normalized_matrix = np.where(non_missing_mask, normalized_matrix, 0)

    return normalized_matrix


def generate_hash(file_path: str, 
                  fit_idx: np.ndarray,
                  trans_idx: np.ndarray) -> str:
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
    hash_input += trans_idx.tobytes()  # Convert fit_idx to bytes

    # Generate and return the hash
    return hashlib.md5(hash_input).hexdigest()


def convert_plink_to_npy(plink_prefix: str,
                         fname: str,
                         fit_idx: np.array,
                         trans_idx: np.array) -> None:
    pedfile = PyPlink(plink_prefix)
    genotypes_array = np.zeros([pedfile.get_nb_samples(), 
                                pedfile.get_nb_markers()], 
                                dtype=np.int8)

    for i, (marker_id, genotypes) in tqdm.tqdm(enumerate(pedfile)):
        genotypes_array[:,i] = genotypes

    # HWE normalization
    normalized_matrix = preprocess_data_matrix(genotypes_array, 
                                               fit_idx,
                                               trans_idx)

    np.save(fname, 
            normalized_matrix)


class PlinkDataset(Dataset):
    """
    PyTorch Dataset for plink formatted genetic datasets.
    """

    def __init__(self, 
                 plink_prefix: str,
                 metadata_path: str,
                 mmap_mode: Optional[str] = None,
                 mode: str = 'genotypes') -> None:
        """
        Initializes the HGDP dataset with configuration parameters.

        Args:
            plink_prefix (str): Path to the plink file + name (excluding extension).
            metadata_path (str): Path to metadata to load
            mmap_mode (Optional[str]): Mode for memory-mapped files. 
                                       Use 'r' for read-only to handle large datasets.
            mode: return processed genotype data or pc transformed data
        """
        super().__init__()
        self.plink_prefix = plink_prefix
        self.mmap_mode = mmap_mode

        # load metadata
        self.metadata = self.load_metadata(metadata_path)

        # extract samples to fit on vs transform
        self.fit_idx, self.trans_idx = self.extract_indices()

        # Create a unique hash for the input arguments
        file_hash = generate_hash(self.plink_prefix, self.fit_idx, self.trans_idx)
        self.npy_cache_file = os.path.join(CACHE_DIR, f".{file_hash}.npy")
        self.pca_cache_file = os.path.join(CACHE_DIR, f".{file_hash}.pca.npy")

        if mode == 'genotypes':
            # if cached file does not exist, create it
            if not os.path.exists(self.npy_cache_file):
                # else load from pedfile and cache
                convert_plink_to_npy(self.plink_prefix,
                                     self.npy_cache_file,
                                     self.fit_idx,
                                     self.trans_idx)
            self.X = np.load(self.npy_cache_file,
                             mmap_mode=self.mmap_mode)

        if mode == 'pca':
            # if PCA file does not exist, create it
            if not os.path.exists(self.pca_cache_file):
                from hail_pca import compute_pca_from_hail
                compute_pca_from_hail(self.plink_prefix,
                                      self.pca_cache_file,
                                      self.metadata,
                                      self.fit_idx,
                                      self.trans_idx)
            
            
            
            #self.X = load_pca_from_hail(self.pca_cache_file, 
            #                            )
            # def load_pca_from_hail(hail_pca_path, merged_metadata, num_pcs):
            #     pca_emb = pd.read_csv(hail_pca_path, sep='\t')
            #     to_return = merged_metadata.merge(pca_emb.set_index('s'), how='left', left_index=True, right_index=True)
            #     to_return = to_return[to_return.columns[to_return.columns.str.startswith('PC')].tolist()].values[:,:num_pcs]
            #     return to_return, None

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        return self.X[index], self.metadata.iloc[index]

    def __len__(self) -> int:
        return len(self.X)
    
    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        sets indices to fit and transform on using metadata.
        """
        raise NotImplementedError
    
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        loads metadata.
        """
        raise NotImplementedError


class HGDPDataset(PlinkDataset):
    """
    PyTorch  Dataset for the Thousand Genomes Project + Human Genome Diversity Project (HGDP) dataset.
    """

    def extract_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
        _filtered_indices = self.metadata[self.metadata[filters].any(axis=1)].index
        filtered_indices = ~self.metadata.index.isin(_filtered_indices)
        related_indices = ~self.metadata['filter_king_related'].values

        to_fit_on = related_indices & filtered_indices
        to_transform_on = (~related_indices) & filtered_indices

        return to_fit_on, to_transform_on

    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        metadata = pd.read_csv(metadata_path)

        # because HGDP metadata is missing first row, we manually add dummy first row
        null_row = pd.DataFrame([{col: np.nan for col in metadata.columns}])
        for _filter in  ["filter_king_related", "filter_pca_outlier", "hard_filtered", "filter_contaminated"]:
            null_row[_filter] = False
        metadata = pd.concat([null_row, metadata], ignore_index=True)
        return metadata.set_index('project_meta.sample_id')