import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as svstack
from tqdm import tqdm


def load_data(paths, data_cfg):
    """
    Load project-specific data, including metadata, PCA scores, and admixture data.

    Args:
        paths (DictConfig): Paths configuration.
        data_cfg (DictConfig): Data configuration.

    Returns:
        Tuple: Merged metadata, PCA scores, and admixture data.
    """
    # Resolve directories
    genotype_dir = Path(paths.genotype_dir)
    admixture_dir = Path(paths.admixture_dir)

    # Load metadata
    metadata_file = genotype_dir / data_cfg.metadata_file
    metadata = pd.read_csv(metadata_file, sep=",")
    metadata = metadata.drop(columns=["Project", "Population", "Genetic_region"], errors="ignore")

    # Load PCA scores
    pca_file = Path(paths.data_dir) / data_cfg.pca_file
    pca_scores = pd.read_csv(pca_file, sep=",")

    # Load admixture metadata
    admixture_data = {
        f: pd.read_csv(admixture_dir / f, sep="\t")
        for f in data_cfg.admixture_files
    }

    return metadata, pca_scores, admixture_data

def load_hgdp_data(data_dir, fname, metadata_file, relatedness_file, filters):
    """
    Load HGDP genotypes, metadata, and relatedness data.

    Args:
        data_dir (str): Base directory containing data files.
        fname (str): Filename for the genotype data.
        metadata_file (str): Path to the metadata file relative to data_dir.
        relatedness_file (str): Path to the relatedness file relative to data_dir.
        filters (list): List of filters to apply to metadata.

    Returns:
        Tuple: Merged metadata, relatedness matrix, genotypes array, and colormap mappings.
    """
    # Resolve paths
    genotype_file = os.path.join(data_dir, fname)
    metadata_file = os.path.join(data_dir, metadata_file)
    relatedness_file = os.path.join(data_dir, relatedness_file)

    # Load genotype data
    genotypes_array = None
    try:
        genotypes_array = np.load(os.path.join(data_dir, '_raw_genotypes.npy'))
    except FileNotFoundError:
        from pyplink import PyPlink
        pedfile = PyPlink(genotype_file)
        genotypes_array = np.zeros([pedfile.get_nb_samples(), pedfile.get_nb_markers()], dtype=np.int8)

        for i, (_, genotypes) in tqdm(enumerate(pedfile), desc="Loading genotypes"):
            genotypes_array[:, i] = genotypes

        np.save(os.path.join(data_dir, '_raw_genotypes.npy'), genotypes_array)

    # Load metadata
    metadata = pd.read_csv(metadata_file, sep=",")
    metadata = metadata.drop(columns=["Project", "Population", "Genetic_region"], errors="ignore")

    # Apply filters
    if filters:
        for filter_name in filters:
            if filter_name in metadata.columns:
                metadata = metadata[~metadata[filter_name]]

    # Load relatedness data
    relatedness = pd.read_csv(relatedness_file, sep="\t", index_col=0)

    # Generate colormap placeholders (customize as needed)
    pop_palette_hgdp_coarse = None
    pop_palette_hgdp_fine = None

    return metadata, relatedness, genotypes_array, (pop_palette_hgdp_coarse, pop_palette_hgdp_fine)

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

def preprocess_data(admixtures_k, data_dir, admixture_dir, genotype_dir, pca_file, metadata_file, relatedness_file, filters):
    """
    Preprocess data for downstream analysis.

    Args:
        admixtures_k (list): List of admixture components (e.g., [2, 3, 4, ...]).
        data_dir (str): Base directory containing data files.
        admixture_dir (str): Directory containing admixture metadata files.
        genotype_dir (str): Directory containing genotype files.
        pca_file (str): Path to the precomputed PCA file.
        metadata_file (str): Path to the metadata file.
        relatedness_file (str): Path to the relatedness file.
        filters (list): List of filters to apply to metadata.

    Returns:
        tuple: PCA embeddings, metadata, indices for fitting and transforming,
               admixture ratios list, and colormap mapping.
    """
    # Step -1: Load HGDP data
    merged_metadata, relatedness, genotypes_array, mapping_info = load_hgdp_data(
        data_dir=data_dir,
        fname="gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.reduced.tsv",
        metadata_file=metadata_file,
        relatedness_file=relatedness_file,
        filters=filters
    )

    # Step 0: Preprocess genotype data
    normalized_matrix, overlap_counts = preprocess_data_matrix(genotypes_array)

    # Filter unrelated samples
    filters = ["filter_pca_outlier", "hard_filtered", "filter_contaminated"]
    _filtered_indices = merged_metadata[merged_metadata[filters].any(axis=1)].index
    filtered_indices = ~merged_metadata.index.isin(_filtered_indices)
    related_indices = ~merged_metadata['filter_king_related'].values

    to_fit_on = related_indices & filtered_indices
    to_transform_on = (~related_indices) & filtered_indices

    # Compute PCA embeddings
    pca_emb, _ = compute_pca_from_hail(pca_file, merged_metadata, 50)

    # Load admixture ratios
    admixture_ratios_list = []
    for n_comps in admixtures_k:
        admixture_file = os.path.join(admixture_dir, f"global.{n_comps}_metadata.tsv")
        admix_ratios = pd.read_csv(admixture_file, sep="\t", header=None)

        # Fill zeros for non-admixture rows
        admixture_ratios_nonzero = admix_ratios.loc[:, 1:n_comps].values
        admixture_ratios = np.zeros((pca_emb.shape[0], admixture_ratios_nonzero.shape[1]))

        index = to_fit_on | to_transform_on
        admixture_ratios[index] = admixture_ratios_nonzero
        admixture_ratios_list.append(admixture_ratios)

    return pca_emb, merged_metadata, to_fit_on, to_transform_on, admixture_ratios_list, mapping_info[1]

import numpy as np

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
