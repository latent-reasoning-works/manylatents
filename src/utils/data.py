import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as svstack
from tqdm import tqdm
from pyplink import PyPlink


from utils.mappings import make_palette_label_order_HGDP

def replace_negative_one_with_nan(array):
    # Replace all occurrences of -1 with np.nan
    return np.where(array == -1, np.nan, array)

def compute_non_missing_overlap(non_missing_mask, recompute=False, save_path="non_missing_overlap.npz"):
    # Check if the result already exists
    if os.path.exists(save_path) and not recompute:
        print("Loading previously computed non-missing overlap matrix...")
        prev_comp = np.load(save_path)['overlap_matrix']
        return prev_comp

    # Convert non-missing mask to sparse format, treating False as 1 and True as 0
    sparse_mask = csr_matrix((~non_missing_mask).astype(int))

    # Initialize a list to store row-wise results
    results = []

    # Iterate over each row with tqdm for progress tracking
    for i in tqdm(range(sparse_mask.shape[0]), desc="Computing row-wise non-missing overlaps"):
        # Compute addition of row `i` with all rows in `sparse_mask`
        replicated_row = svstack([sparse_mask[i]] * sparse_mask.shape[0])

        # Count non-zero entries for each pair (row i + row j)
        nonzero_counts = (replicated_row+sparse_mask).getnnz(axis=1)

        # Append the non-zero counts as a sparse row to results
        results.append(nonzero_counts)

    # Stack all the result rows to form the final matrix
    final_result = np.vstack(results)
    final_result = len(non_missing_mask[0]) - final_result
    np.savez_compressed(save_path, overlap_matrix=final_result)

    return final_result

def hwe_normalize(genotypes_array):

    # Compute allele frequencies, ignoring NaNs
    allele_freqs = np.nanmean(genotypes_array / 2, axis=0)  # p = mean allele frequency

    # Center the matrix by subtracting 2 * allele frequency for each SNP
    centered_matrix = genotypes_array - 2 * allele_freqs

    # Compute Hardy-Weinberg variance for each SNP, avoiding division by zero
    hwe_variance = 2 * allele_freqs * (1 - allele_freqs)
    hwe_variance[hwe_variance == 0] = 1  # Avoid division by zero for monomorphic SNPs

    # Normalize each SNP by Hardy-Weinberg variance
    normalized_matrix = centered_matrix / np.sqrt(hwe_variance)
    return normalized_matrix

def preprocess_data_matrix(genotypes_array, recompute_overlap_counts=False):
    
    # Compute hwe normalized matrix
    genotypes_array = replace_negative_one_with_nan(genotypes_array)
    normalized_matrix = hwe_normalize(genotypes_array)

    # Create a mask for non-missing values
    non_missing_mask = ~np.isnan(genotypes_array)

    # Replace NaNs in the normalized matrix with zeros for compatibility with matrix multiplication
    normalized_matrix = np.where(non_missing_mask, normalized_matrix, 0)

    # speeds up computation by exploiting sparsity
    overlap_counts = compute_non_missing_overlap(non_missing_mask, recompute_overlap_counts)
    assert np.allclose(overlap_counts[:2], np.dot(non_missing_mask[0:2].astype(int), non_missing_mask.T))
    return normalized_matrix, overlap_counts

def compute_pca_from_hail(hail_pca_path, merged_metadata, num_pcs):
    pca_emb = pd.read_csv(hail_pca_path, sep='\t')
    to_return = merged_metadata.merge(pca_emb.set_index('s'), how='left', left_index=True, right_index=True)
    to_return = to_return[to_return.columns[to_return.columns.str.startswith('PC')].tolist()].values[:,:num_pcs]
    return to_return, None


def calculate_maf(genotypes):
    """
    Calculate minor allele frequency (MAF) for each SNP (column) in the genotype matrix.

    Args:
        genotypes (numpy.ndarray): Genotype matrix of shape (num_samples, num_SNPs) with values 0, 1, or 2.
        
    Returns:
        numpy.ndarray: MAF for each SNP.
    """
    allele_sum = np.sum(genotypes, axis=0)
    num_samples = genotypes.shape[0]
    
    # Frequency of the minor allele
    maf = allele_sum / (2 * num_samples)
    
    # Ensure MAF is for the minor allele (MAF is the smaller of the two allele frequencies)
    maf = np.minimum(maf, 1 - maf)
    
    return maf

def maf_scale(genotypes):
    """
    Apply MAF scaling to genotype data.

    Args:
        genotypes (numpy.ndarray): Genotype data array with values 0, 1, or 2 (shape: (num_samples, num_SNPs)).
        
    Returns:
        numpy.ndarray: MAF-scaled genotype data.
    """
    maf = calculate_maf(genotypes)  # Calculate MAF for each SNP
    scaled_data = (genotypes - 2 * maf) / np.sqrt(2 * maf * (1 - maf))  # Apply MAF scaling
    
    return scaled_data

def load_hgdp_data(genotype_dir, metadata_file, genotype_prefix, filters):
    """
    Load HGDP genotypes and metadata.

    Args:
        genotype_dir (str): Directory containing genotype files.
        metadata_file (str): Path to the metadata file relative to genotype_dir.
        genotype_prefix (str): Prefix of the genotype files.
        filters (list): List of filters to apply to metadata.

    Returns:
        Tuple: Merged metadata, genotypes array, and colormap mappings.
    """
    # Resolve paths
    metadata_file_path = os.path.join(genotype_dir, metadata_file)
    required_files = ["bed", "bim", "fam"]

    # Check for required genotype files
    missing_files = [
        ext for ext in required_files if not os.path.exists(f"{genotype_dir}/{genotype_prefix}.{ext}")
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing required genotype files: {', '.join(missing_files)}")

    # Load genotype data
    try:
        genotypes_array = np.load(os.path.join(genotype_dir, "_raw_genotypes.npy"))
    except FileNotFoundError:
        pedfile = PyPlink(os.path.join(genotype_dir, genotype_prefix))
        genotypes_array = np.zeros((pedfile.get_nb_samples(), pedfile.get_nb_markers()), dtype=np.int8)
        for i, (_, genotypes) in tqdm(enumerate(pedfile), desc="Loading genotypes"):
            genotypes_array[:, i] = genotypes
        np.save(os.path.join(genotype_dir, "_raw_genotypes.npy"), genotypes_array)

    # Load metadata
    metadata = pd.read_csv(metadata_file_path, sep=",")
    metadata.columns = metadata.columns.str.strip()

    # Apply filters
    for filter_name in filters:
        if filter_name in metadata.columns:
            metadata = metadata[~metadata[filter_name]]

    # Generate colormaps
    if "Population" not in metadata.columns or "Superpopulation" not in metadata.columns:
        raise KeyError("Metadata must contain 'Population' and 'Superpopulation' columns.")
    populations = metadata["Population"].values
    superpopulations = metadata["Superpopulation"].values
    pop_palette_hgdp_coarse, pop_palette_hgdp_fine, _, _ = make_palette_label_order_HGDP(
        populations, superpopulations
    )

    return metadata, genotypes_array, (pop_palette_hgdp_coarse, pop_palette_hgdp_fine)

def preprocess_data(admixtures_k, data_dir, admixture_dir, genotype_dir, metadata_file, genotype_prefix, filters):
    """
    Preprocess data for downstream analysis.

    Args:
        admixtures_k (list): List of admixture components (K values).
        data_dir (str): Base directory containing data files.
        admixture_dir (str): Directory containing admixture metadata files.
        genotype_dir (str): Directory containing genotype files.
        metadata_file (str): Path to the metadata file relative to genotype_dir.
        genotype_prefix (str): Prefix for genotype files.
        filters (list): List of filters to apply to metadata.

    Returns:
        tuple: PCA embeddings, metadata, indices for fitting and transforming,
               admixture ratios list, and colormap mapping.
    """
    # Load HGDP data
    metadata, genotypes_array, colormaps = load_hgdp_data(
        genotype_dir=genotype_dir,
        metadata_file=metadata_file,
        genotype_prefix=genotype_prefix,
        filters=filters
    )

    # Preprocess genotype data
    normalized_matrix, overlap_counts = preprocess_data_matrix(genotypes_array)

    # Filter indices for unrelated and uncontaminated samples
    _filtered_indices = metadata[metadata[filters].any(axis=1)].index
    filtered_indices = ~metadata.index.isin(_filtered_indices)
    related_indices = ~metadata["filter_king_related"]

    fit_idx = related_indices & filtered_indices
    transform_idx = ~related_indices & filtered_indices

    # Compute PCA embeddings
    pca_file_path = os.path.join(data_dir, "pca_scores_hailcomputed.csv")
    pca_emb, _ = compute_pca_from_hail(pca_file_path, metadata, 50)

    # Load admixture ratios
    admixture_ratios_list = []
    for k in admixtures_k:
        admixture_file_path = os.path.join(admixture_dir, f"global.{k}_metadata.tsv")
        admixture_data = pd.read_csv(admixture_file_path, sep="\t", header=None)
        admixture_ratios = np.zeros((pca_emb.shape[0], k))

        indices_to_fill = fit_idx | transform_idx
        admixture_ratios[indices_to_fill] = admixture_data.iloc[:, 1:k + 1].values
        admixture_ratios_list.append(admixture_ratios)

    return pca_emb, metadata, fit_idx, transform_idx, admixture_ratios_list, colormaps

def load_preprocessed_data(admixtures_k, data_dir, admixture_dir, genotype_dir, metadata_file, genotype_prefix, filters):
    """
    Wrapper for preprocessing data and preparing for hyperparameter sweeps.

    Args:
        admixtures_k (list): List of admixture components (K values).
        data_dir (str): Directory containing data files.
        admixture_dir (str): Directory containing admixture metadata files.
        genotype_dir (str): Directory containing genotype files.
        metadata_file (str): Path to the metadata file relative to genotype_dir.
        genotype_prefix (str): Prefix for genotype files.
        filters (list): List of filters to apply to metadata.

    Returns:
        tuple: PCA embeddings, metadata, fit/transform indices, admixture ratios list, and colormap mappings.
    """
    return preprocess_data(
        admixtures_k=admixtures_k,
        data_dir=data_dir,
        admixture_dir=admixture_dir,
        genotype_dir=genotype_dir,
        metadata_file=metadata_file,
        genotype_prefix=genotype_prefix,
        filters=filters
    )
